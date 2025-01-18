import cv2
from ultralytics import YOLO
import os
import torch
import sys

def detect_objects(frame, model, device):
    results = model(frame, device=device, stream=True)
    detections = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            dets = {
                'x1': int(r[0]),
                'y1': int(r[1]),
                'x2': int(r[2]),
                'y2': int(r[3]),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0])
            }
            detections.append(dets)
    
    return detections

def save_clip(video_path, start_frame, end_frame, output_folder, clip_index):
    import subprocess

    output_path = os.path.join(output_folder, f"clip_{clip_index}.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-ss', f'{start_time}',
        '-t', f'{duration}',
        '-c:v', 'copy',
        '-c:a', 'copy',
        output_path
    ]

    subprocess.run(command)

def split_by_concept(video_path, output_folder, detection_duration_threshold, max_clip_duration, model, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = None
    clip_index = 0
    detection_count = 0

    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_objects(frame, model, device)
        
        if len(detections) > 0:
            if start_frame is None:
                start_frame = frame_number
            detection_count += 1
        else:
            if start_frame is not None:
                detected_duration = detection_count / fps
                if detected_duration >= detection_duration_threshold:
                    end_frame = min(frame_number - 1, int(start_frame + max_clip_duration * fps))
                    save_clip(video_path, start_frame, end_frame, output_folder, clip_index)
                    clip_index += 1
                
                start_frame = None
                detection_count = 0

    if start_frame is not None:
        detected_duration = detection_count / fps
        if detected_duration >= detection_duration_threshold:
            end_frame = min(total_frames - 1, int(start_frame + max_clip_duration * fps))
            save_clip(video_path, start_frame, end_frame, output_folder, clip_index)

    cap.release()
