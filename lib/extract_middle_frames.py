import os
import cv2
from glob import glob
def extract_middle_frames(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    video_files = glob(os.path.join(source_dir, '**', '*.mp4'), recursive=True)

    for video_file in video_files:
        relative_path = os.path.relpath(video_file, start=source_dir)
        dest_folder = os.path.join(dest_dir, os.path.dirname(relative_path))
        
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        
        success, frame = cap.read()
        if success:
            dest_file = os.path.join(dest_folder, f'{os.path.splitext(os.path.basename(video_file))[0]}_middle_frame.jpg')
            cv2.imwrite(dest_file, frame)
        
        cap.release()
