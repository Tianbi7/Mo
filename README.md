# Mo
墨意
import cv2
import mediapipe as mp
import numpy as np
import time
import os

save_dir = "output_calligraphy"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_counter = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
hand_model_path = os.path.join(script_dir, 'hand_landmarker.task')

with open(hand_model_path, 'rb') as f:
    hand_model_content = f.read()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=hand_model_content),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

hand_detector = HandLandmarker.create_from_options(hand_options)

def init_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    return cap
            cap.release()
    return None

cap = init_camera()
if cap is None:
    print("ERROR: Cannot open camera")
    exit(1)

w, h = 640, 480

ink_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
trace_points = []
MAX_TRACE_POINTS = 15

last_time = time.time()
color_idx = 0
ink_colors = [(0,0,0), (0,0,200), (120,120,120)]
was_pinch = False
was_fist = False
fist_frame_count = 0
palm_open_frame_count = 0
ink_droplets = []

def get_point(landmark, img_w, img_h):
    return int(landmark.x * img_w), int(landmark.y * img_h)

def ink_render(canvas, points, speed):
    if len(points) < 2:
        return canvas
    base_radius = max(3, int(15 - speed * 0.08))
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i+1]
        local_radius = max(2, int(base_radius * (0.8 + np.random.rand() * 0.2)))
        cv2.line(canvas, pt1, pt2, ink_colors[color_idx], thickness=local_radius)
    canvas = cv2.GaussianBlur(canvas, (3,3), sigmaX=0.5)
    return canvas

def judge_palm_open(landmarks):
    tip_ids = [4,8,12,16,20]
    mcp_ids = [2,5,9,13,17]
    dist_sum = 0
    for t, m in zip(tip_ids, mcp_ids):
        t_pt = np.array([landmarks[t].x, landmarks[t].y])
        m_pt = np.array([landmarks[m].x, landmarks[m].y])
        dist_sum += np.linalg.norm(t_pt - m_pt)
    return dist_sum / 5 > 0.10

def judge_fist(landmarks):
    tip_ids = [4,8,12,16,20]
    mcp_ids = [2,5,9,13,17]
    dist_sum = 0
    for t, m in zip(tip_ids, mcp_ids):
        t_pt = np.array([landmarks[t].x, landmarks[t].y])
        m_pt = np.array([landmarks[m].x, landmarks[m].y])
        dist_sum += np.linalg.norm(t_pt - m_pt)
    return dist_sum / 5 < 0.06

def get_thumb_index_dist(landmarks):
    t = landmarks[4]
    i = landmarks[8]
    return np.sqrt((t.x-i.x)**2 + (t.y-i.y)**2)

def is_right_hand(results, hand_index):
    if results.handedness:
        return results.handedness[hand_index][0].category_name == 'Right'
    return True

def main():
    global ink_canvas, trace_points, was_pinch, was_fist, fist_frame_count, palm_open_frame_count, ink_droplets, color_idx, save_counter
    
    hand_detected = False
    is_writing = False
    idle_start = time.time()
    last_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Air Calligraphy", frame)
                cv2.imshow("Ink Canvas", ink_canvas)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue
            
            frame = cv2.flip(frame, 1)
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            try:
                results = hand_detector.detect_for_video(mp_image, timestamp_ms)
            except:
                results = None
            
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            
            index_tip = None
            hand_detected = False
            current_palm_open = False
            current_fist = False
            right_hand_lm = None
            
            if results and results.hand_landmarks:
                hand_detected = True
                
                for i, hand_lm in enumerate(results.hand_landmarks):
                    if is_right_hand(results, i):
                        right_hand_lm = hand_lm
                        break
                
                if right_hand_lm is None and results.hand_landmarks:
                    right_hand_lm = results.hand_landmarks[0]
                
                if right_hand_lm:
                    landmarks = right_hand_lm
                    
                    for id, lm in enumerate(landmarks):
                        cx = int(lm.x * w)
                        cy = int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    
                    connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]]
                    for conn in connections:
                        x1, y1 = get_point(landmarks[conn[0]], w, h)
                        x2, y2 = get_point(landmarks[conn[1]], w, h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    index_tip = get_point(landmarks[8], w, h)
                    cv2.circle(frame, index_tip, 8, (0, 0, 255), -1)
                    
                    current_palm_open = judge_palm_open(landmarks)
                    current_fist = judge_fist(landmarks)
                    thumb_idx_dist = get_thumb_index_dist(landmarks)
                    
                    if thumb_idx_dist < 0.04 and not was_pinch:
                        color_idx = (color_idx + 1) % len(ink_colors)
                        was_pinch = True
                    elif thumb_idx_dist >= 0.04:
                        was_pinch = False
                    
                    if current_fist:
                        fist_frame_count += 1
                        is_writing = False
                    else:
                        fist_frame_count = 0
                    
                    if current_fist and fist_frame_count > 5 and not was_fist:
                        ink_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                        trace_points.clear()
                        ink_droplets.clear()
                        fist_frame_count = 0
                        is_writing = False
                    
                    was_fist = current_fist
                    
                    if current_palm_open:
                        palm_open_frame_count += 1
                        is_writing = False
                        if palm_open_frame_count > 3:
                            wrist_pos = get_point(landmarks[0], w, h)
                            ink_droplets.append([wrist_pos[0], wrist_pos[1], 0, 40 + np.random.randint(20), 2, 0.8])
                            palm_open_frame_count = 0
                    else:
                        palm_open_frame_count = 0
                
                if index_tip is not None and not current_palm_open and not current_fist:
                    is_writing = True
                    
                    if len(trace_points) == 0:
                        trace_points.append(index_tip)
                    else:
                        last_pt = trace_points[-1]
                        dist = np.sqrt((index_tip[0]-last_pt[0])**2 + (index_tip[1]-last_pt[1])**2)
                        if dist > 1:
                            trace_points.append(index_tip)
                            if len(trace_points) > MAX_TRACE_POINTS:
                                trace_points.pop(0)
                    
                    if len(trace_points) >= 2:
                        speed = 0
                        if dt > 0:
                            speed = np.sqrt((trace_points[-1][0]-trace_points[-2][0])**2 + 
                                           (trace_points[-1][1]-trace_points[-2][1])**2) / dt
                        ink_canvas = ink_render(ink_canvas, trace_points, speed)
            else:
                trace_points.clear()
                was_fist = False
                is_writing = False
            
            for droplet in ink_droplets[:]:
                droplet[2] += droplet[4]
                droplet[5] -= 0.015
                if droplet[5] <= 0:
                    ink_droplets.remove(droplet)
                else:
                    for r in range(int(droplet[2]), 0, -2):
                        cv2.circle(ink_canvas, (droplet[0], droplet[1]), r, (0, 0, 0), thickness=-1)
            
            if time.time() - idle_start > 5 and np.sum(255 - ink_canvas) > 500:
                save_path = os.path.join(save_dir, f"calli_{save_counter}.png")
                cv2.imwrite(save_path, ink_canvas)
                save_counter += 1
                idle_start = time.time()
            
            ink_gray = cv2.cvtColor(ink_canvas, cv2.COLOR_BGR2GRAY)
            _, ink_mask = cv2.threshold(ink_gray, 250, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(ink_mask)
            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv2.bitwise_and(ink_canvas, ink_canvas, mask=ink_mask)
            frame_with_ink = cv2.add(bg, fg)
            
            color_text = ["黑墨", "红墨", "灰墨"][color_idx]
            cv2.putText(frame_with_ink, f"墨色:{color_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame_with_ink, f"手: {'检测到' if hand_detected else '未检测'}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(frame_with_ink, f"状态: {'书写中' if is_writing else ('握拳清除' if current_fist else '空闲')}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame_with_ink, "食指书写 | 握拳清空 | 张开墨滴 | 捏合切换", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame_with_ink, "按Q退出", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            cv2.imshow("Air Calligraphy", frame_with_ink)
            cv2.imshow("Ink Canvas", ink_canvas)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()