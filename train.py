from ultralytics import YOLO

# Cargar modelo base
model = YOLO("yolov8n.pt")

# Entrenar con el dataset
results = model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="ppe_detector",
    project="runs/detect"
)

print("Entrenamiento completado")
print("Modelo guardado en: runs/detect/ppe_detector/weights/best.pt")
