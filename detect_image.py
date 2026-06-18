"""
Detección de EPP en imágenes estáticas
"""
from ultralytics import YOLO
import argparse
import os


def detect_image(image_path, model_path="runs/detect/ppe_detector/weights/best.pt"):
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encontró la imagen: {image_path}")
        return

    # Cargar modelo
    print(f"📦 Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # Realizar detección
    print(f"🔍 Analizando imagen: {image_path}")
    results = model(image_path)

    # Mostrar resultados
    results[0].show()

    # Guardar resultado
    output_path = "resultado_deteccion.jpg"
    results[0].save(output_path)
    print(f"✓ Imagen guardada en: {output_path}")

    # Mostrar detecciones
    print("\n📊 Detecciones encontradas:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        print(f"   - {name}: {conf:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección de EPP en imágenes")
    parser.add_argument("image", help="Ruta de la imagen a analizar")
    parser.add_argument(
        "--model", default="runs/detect/ppe_detector/weights/best.pt", help="Ruta del modelo")

    args = parser.parse_args()
    detect_image(args.image, args.model)
