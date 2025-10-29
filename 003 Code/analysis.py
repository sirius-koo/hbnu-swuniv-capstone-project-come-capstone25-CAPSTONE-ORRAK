import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
import subprocess
import re

# ================== 각도 계산 ==================
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)
    cos_angle = np.dot(ba_norm, bc_norm)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# ================== COM 계산 ==================
segment_params = {
    0: {'proximal': 11, 'distal': 13, 'mass_ratio': 0.028},
    1: {'proximal': 13, 'distal': 15, 'mass_ratio': 0.016},
    2: {'proximal': 12, 'distal': 14, 'mass_ratio': 0.028},
    3: {'proximal': 14, 'distal': 16, 'mass_ratio': 0.016},
    4: {'proximal': 23, 'distal': 25, 'mass_ratio': 0.10},
    5: {'proximal': 25, 'distal': 27, 'mass_ratio': 0.0465},
    6: {'proximal': 24, 'distal': 26, 'mass_ratio': 0.10},
    7: {'proximal': 26, 'distal': 28, 'mass_ratio': 0.0465},
    8: {'proximal': 11, 'distal': 12, 'mass_ratio': 0.14},
    9: {'proximal': 23, 'distal': 24, 'mass_ratio': 0.14},
}

def compute_segment_com(lms, proximal_idx, distal_idx):
    p = lms[proximal_idx]
    d = lms[distal_idx]
    return np.array([(p.x + d.x)/2, (p.y + d.y)/2, (p.z + d.z)/2])

def compute_total_com(lms):
    total_mass = 0
    weighted_sum = np.zeros(3)
    for seg in segment_params.values():
        proximal, distal, mass = seg['proximal'], seg['distal'], seg['mass_ratio']
        if proximal >= len(lms) or distal >= len(lms):
            continue
        seg_com = compute_segment_com(lms, proximal, distal)
        weighted_sum += seg_com * mass
        total_mass += mass
    return weighted_sum / total_mass if total_mass > 0 else None

# ================== 영상 분석 ==================
def analyze_video(video_path, result_path, ffmpeg_path="ffmpeg"):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # --- FFmpeg로 픽셀 기반 회전 처리 ---
    # 기존코드
    # normalized_video_path = video_path.replace(".mp4", f"_normalized_{uuid.uuid4().hex}.mp4")

    # 새 코드
    base_name = os.path.splitext(video_path)[0] # [0]은 확장자를 제외한 경로/파일명 (예: uploads\IMG_0922)
    normalized_video_path = f"{base_name}_normalized_{uuid.uuid4().hex}.mp4"

    # --- FFprobe로 회전 값 직접 읽기 (더 안정적인 방법) ---
    print("=========================================")
    print("     [DEBUG] FFprobe 회전 로직 시작")
    print("=========================================")
    
    # ffmpeg_path에서 ffprobe 경로 추론
    ffprobe_path = "ffprobe" # 기본값 (PATH에 잡혀있을 경우)
    if ffmpeg_path != "ffmpeg":
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        # 윈도우 환경(.exe)을 기본으로 가정
        ffprobe_exe_path = os.path.join(ffmpeg_dir, "ffprobe.exe")
        if os.path.exists(ffprobe_exe_path):
            ffprobe_path = ffprobe_exe_path
        else:
            # .exe가 아닌 경우 (Linux/Mac 또는 경로에 .exe가 생략된 경우)
            ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")

    print(f"[DEBUG] ffprobe 경로: {ffprobe_path}")

    rotate_code = 0
    try:
        # FFprobe 명령어: 비디오 스트림(v:0)에서 'rotate' 태그 값만 출력
        ffprobe_command = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            # 기존 "-show_entries", "stream_tags=rotate",
            "-show_entries", "stream_side_data=rotation",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        # text=True, encoding='utf-8'로 인코딩을 명시
        probe = subprocess.run(ffprobe_command, capture_output=True, text=True, encoding='utf-8', check=True)
        
        stdout = probe.stdout.strip()
        print(f"[DEBUG] FFprobe raw output: '{stdout}'") # (1) ffprobe 출력값
        
        if stdout:
            # ffprobe가 '-90.00' 또는 '-90' 등을 반환
            rotate_code = int(float(stdout)) 
        else:
            print("[DEBUG] FFprobe 출력값이 비어있음 (회전 정보 없음)")

    except FileNotFoundError:
        print(f"[DEBUG] FFprobe를 찾을 수 없습니다: {ffprobe_path}")
    except Exception as e:
        print(f"[DEBUG] FFprobe 실행 중 오류 (영상에 회전 태그가 없을 수 있음): {e}")
    
    print(f"[DEBUG] rotate_code (파싱 후): {rotate_code}") # (2)

    # --- 메타데이터의 음수 각도(예: -90)를 양수(예: 270)로 변환 ---
    if rotate_code == -90:
        rotate_code = 270
    elif rotate_code == -180:
        rotate_code = 180
    elif rotate_code == -270:
        rotate_code = 90
        
    print(f"[DEBUG] rotate_code (음수 변환 후): {rotate_code}") # (3)

    transpose_flag = None 
    
    if rotate_code == 90:
        transpose_flag = "transpose=1"  # 시계방향 90도
    elif rotate_code == 180:
        transpose_flag = "transpose=2,transpose=2"  # 180도
    elif rotate_code == 270:
        transpose_flag = "transpose=2"  # 시계반대방향 90도
        
    print(f"[DEBUG] transpose_flag (최종): {transpose_flag}") # (4)
    print("=========================================")

    if transpose_flag:
        print("[DEBUG] 'if transpose_flag:' 실행 (회전 O)") # (6)
        subprocess.run([
            ffmpeg_path, "-y", 
            "-hwaccel", "none", # 하드웨어 가속 비활성화 !!!!!!!
            "-i", video_path, 
            "-vf", transpose_flag,
            "-c:a", "copy", normalized_video_path
        ], check=True)
    else:
        print("[DEBUG] 'else:' 실행 (회전 X, 복사)") # (7)
        subprocess.run([
            ffmpeg_path, "-y", "-i", video_path, "-c", "copy", normalized_video_path
        ], check=True)

    # --- OpenCV 분석 ---
    cap = cv2.VideoCapture(normalized_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    temp_path = result_path.replace(".mp4", f"_cv_temp_{uuid.uuid4().hex}.mp4")
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # (↓↓↓ 루프 시작 전에 변수 2개 추가 ↓↓↓)
    frame_counter = 0
    last_known_results = None

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # (↓↓↓ 카운터 증가 및 pose.process() 위치 이동 ↓↓↓)
            frame_counter += 1

            # 기존
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = pose.process(frame_rgb)

            # 새 코드: 3프레임당 1번만 mediapipe 실행
            if frame_counter % 3 == 0: 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks: # 유효한 결과만 저장
                    last_known_results = results
            else:
                # 3프레임 중 2프레임은 기존 결과 사용
                results = last_known_results

            r_angle, l_angle = None, None
            r_status, l_status = "---", "---"

            # 기존
            # if results.pose_landmarks:
            #     lms = results.pose_landmarks.landmark
            
            if results and results.pose_landmarks: 
                lms = results.pose_landmarks.landmark
                
                r_shoulder, r_elbow, r_wrist = lms[11], lms[13], lms[15]
                l_shoulder, l_elbow, l_wrist = lms[12], lms[14], lms[16]
                
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                r_status = "Stable" if r_angle > 95 else "Unstable"
                l_status = "Stable" if l_angle > 95 else "Unstable"

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                )

                r_color = (0,255,0) if r_status=="Stable" else (0,0,255)
                l_color = (0,255,0) if l_status=="Stable" else (0,0,255)

                cv2.circle(frame, (int(r_elbow.x*width), int(r_elbow.y*height)), 10, r_color, -1)
                cv2.circle(frame, (int(l_elbow.x*width), int(l_elbow.y*height)), 10, l_color, -1)

                com = compute_total_com(lms)
                pts_ids = [19, 20, 32, 31]
                if all(i < len(lms) for i in pts_ids):
                    pts = np.array([[int(lms[i].x*width), int(lms[i].y*height)] for i in pts_ids], np.int32)
                    inside = -1
                    if com is not None:
                        com_px = (int(com[0]*width), int(com[1]*height))
                        inside = cv2.pointPolygonTest(pts, com_px, False)
                        cv2.circle(frame, com_px, 10, (0,255,255), -1)
                    color = (0,255,0) if inside>0 else (0,0,255)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            r_text = f"Right Elbow: {f'{r_angle:.1f}' if r_angle else '---'} ({r_status})"
            l_text = f"Left Elbow:  {f'{l_angle:.1f}' if l_angle else '---'} ({l_status})"
            for j, line in enumerate([r_text,l_text]):
                cv2.putText(frame, line, (10,30 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

            out.write(frame)

    cap.release()
    out.release()
    os.remove(normalized_video_path)

    # --- 최종 변환 ---
    subprocess.run([
        ffmpeg_path, "-y", "-i", temp_path, "-c:v", "libx264", 
        # 기존: 
        "-preset", "fast", 
        # "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", result_path
    ], check=True)
    os.remove(temp_path)
