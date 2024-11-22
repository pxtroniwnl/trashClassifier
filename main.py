from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(data="/dataset-resized", epochs=30, imgsz=240)

