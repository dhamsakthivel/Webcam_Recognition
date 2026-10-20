#!/usr/bin/env python3
"""Export a local COCO YOLO ONNX model for optional phone/bottle/person detection."""
import json
import shutil
from pathlib import Path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install export dependencies first: python -m pip install --upgrade ultralytics") from error

    model = YOLO("yolo11n.pt")
    exported_onnx = Path(model.export(format="onnx", imgsz=640, simplify=False, opset=17))
    destination = Path("models")
    destination.mkdir(exist_ok=True)
    shutil.copy2(exported_onnx, destination / "general-yolo.onnx")
    names = model.names
    (destination / "general-yolo.labels.json").write_text(
        json.dumps([names[index] for index in range(len(names))]), encoding="utf-8"
    )
    print("Local COCO detector exported to models/general-yolo.onnx")


if __name__ == "__main__":
    main()
