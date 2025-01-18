from ultralytics import YOLO
from pathlib import Path

def train_yolo11_model(dataset_path, epochs, model):
    model = YOLO(model)

    path_parts = Path(dataset_path).parts
    project = str(Path(*path_parts[:-1]))
    name = f"{path_parts[-1]}_done"

    results = model.train(
        data=f"{dataset_path}/data.yaml",
        project=project,
        name=name,
        epochs=epochs,
        imgsz=640
    )
    return results
