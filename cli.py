"""Multi-tool script.

Usage:
    cli.py describe <folder_path> [--prompt=<prompt>] [--image-glob-pattern=<image_glob_pattern>]
    cli.py extract-middle-frame <source_dir> <dest_dir>
    cli.py split-by-concept <video_path> [--detection-duration-threshold=<detection_duration_threshold>] [--max-clip-duration=<max_clip_duration>] [--yolo-model-path=<yolo_model_path>]
    cli.py train-yolo11 <dataset_path> [--epochs=<epochs>] [--yolo-model-path=<model_path>]

Options:
    -h --help       Show this screen.
    -p <prompt> --prompt=<prompt>         Prompt to use for describing images [default: Describe image]
    -g <image_glob_pattern> --image-glob-pattern=<image_glob_pattern>   Image glob pattern to match images in the folder [default: **/*.[jpJp][pnNP]g]
    -t <detection_duration_threshold> --detection-duration-threshold=<detection_duration_threshold>   Minimum duration of a detection for splitting video [default: 3]
    -m <max_clip_duration> --max-clip-duration=<max_clip_duration>     Maximum duration for each video clip [default: 30]
    -y <yolo_model_path> --yolo-model-path=<yolo_model_path>           Path to the YOLO11 model [default: models/yolo11n.pt]
    -e <epochs> --epochs=<epochs>                                      Number of epochs for training the YOLO model [default: 50]
"""
from docopt import docopt

def main():
    args = docopt(__doc__)

    if args['describe']:
        import os
        folder_path = args['<folder_path>']
        prompt = args['--prompt']
        image_glob_pattern = os.path.join(folder_path, args['--image-glob-pattern'])
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
            torch_dtype="auto",
            device_map="auto"
        )
        model.to(device)

        from lib.describe_images import describe_images
        describe_images(folder_path, prompt, image_glob_pattern, processor, model)
    
    elif args['extract-middle-frame']:
        source_dir = args['<source_dir>']
        dest_dir = args['<dest_dir>']
        from lib.extract_middle_frames import extract_middle_frames
        extract_middle_frames(source_dir, dest_dir)

    elif args['split-by-concept']:
        video_path = args['<video_path>']
        detection_duration_threshold = int(args['--detection-duration-threshold'])
        max_clip_duration = int(args['--max-clip-duration'])
        yolo_model_path = args['--yolo-model-path']

        import torch
        from ultralytics import YOLO

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(yolo_model_path)
        model.to(device)

        output_folder = f"{video_path}_clips"
        from lib.split_by_concept import split_by_concept
        split_by_concept(video_path, output_folder, detection_duration_threshold, max_clip_duration, model, device)
    
    elif args['train-yolo11']:
        dataset_path = args['<dataset_path>']
        epochs = int(args['--epochs'])
        model_path = args['--yolo-model-path']
        from lib.train_yolo11 import train_yolo11_model
        train_yolo11_model(dataset_path, epochs, model_path)

if __name__ == "__main__":
    main()
