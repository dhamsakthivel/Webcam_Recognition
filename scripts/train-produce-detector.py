#!/usr/bin/env python3
"""Train Universidad de León YOLO annotations and export a local web ONNX model."""
import argparse
import json
import shutil
from pathlib import Path


def find_data_yaml(dataset_root: Path) -> Path:
    matches = list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("dataset.yaml"))
    if not matches:
        raise FileNotFoundError(f"No YOLO data.yaml found below {dataset_root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/datasets/unileon-yolo-detection"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default="yolo11n.pt")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install training dependencies first: python -m pip install --upgrade ultralytics") from error

    yaml_file = args.data if args.data.suffix in {".yaml", ".yml"} else find_data_yaml(args.data)
    model = YOLO(args.base_model)
    result = model.train(data=str(yaml_file), epochs=args.epochs, imgsz=args.imgsz)
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    trained = YOLO(best_weights)
    exported_onnx = Path(trained.export(format="onnx", imgsz=args.imgsz, simplify=False, opset=17))

    destination = Path("models")
    destination.mkdir(exist_ok=True)
    shutil.copy2(exported_onnx, destination / "produce-yolo.onnx")
    names = trained.names
    labels = [names[index] for index in range(len(names))]
    (destination / "produce-yolo.labels.json").write_text(json.dumps(labels), encoding="utf-8")
    print("Local produce detector exported to models/produce-yolo.onnx")


if __name__ == "__main__":
    main()
