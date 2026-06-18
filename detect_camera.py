"""
Detección de EPP en tiempo real con cámara
"""
from ultralytics import YOLO
import cv2


def detect_camera(model_path="runs/detect/ppe_detector/weights/best.pt"):
    # Cargar modelo
    print(f"📦 Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # Abrir cámara
    print("📷 Iniciando cámara...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return

    print("✓ Cámara iniciada. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección
        results = model(frame, verbose=False)

        # Dibujar resultados en el frame
        annotated_frame = results[0].plot()

        # Mostrar frame
        cv2.imshow("Detección EPP - Presiona 'q' para salir", annotated_frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Cámara cerrada")


if __name__ == "__main__":
    detect_camera()
