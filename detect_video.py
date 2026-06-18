"""
Detección de EPP en videos
"""
from ultralytics import YOLO
import cv2
import argparse
import os


def detect_video(video_path, model_path="runs/detect/ppe_detector/weights/best.pt"):
    # Verificar que el video existe
    if not os.path.exists(video_path):
        print(f"❌ Error: No se encontró el video: {video_path}")
        return

    # Cargar modelo
    print(f"📦 Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # Abrir video
    print(f"🎬 Procesando video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir el video")
        return

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crear video de salida
    output_path = "resultado_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"⏱️ FPS: {fps}, Resolución: {width}x{height}")
    print("Procesando... Presiona 'q' para cancelar")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección
        results = model(frame, verbose=False)

        # Dibujar resultados
        annotated_frame = results[0].plot()

        # Escribir frame
        out.write(annotated_frame)

        # Mostrar progreso
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Procesados {frame_count} frames...")

        # Mostrar preview (opcional)
        cv2.imshow("Procesando video - Presiona 'q' para cancelar",
                   annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠️ Cancelado por el usuario")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✓ Video guardado en: {output_path}")
    print(f"✓ Total frames procesados: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detección de EPP en videos")
    parser.add_argument("video", help="Ruta del video a analizar")
    parser.add_argument(
        "--model", default="runs/detect/ppe_detector/weights/best.pt", help="Ruta del modelo")

    args = parser.parse_args()
    detect_video(args.video, args.model)
