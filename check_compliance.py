"""
Verificación de cumplimiento de EPP
Detecta si una persona tiene el equipo de protección completo
"""
from ultralytics import YOLO
import cv2

# Clases del modelo
CLASSES = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}

# EPP requerido
EPP_REQUERIDO = ['Hardhat', 'Safety Vest', 'Mask']
EPP_FALTANTE = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']


def check_compliance(model_path="runs/detect/ppe_detector/weights/best.pt"):
    # Cargar modelo
    print(f"📦 Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # Abrir cámara
    print("📷 Iniciando cámara...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return

    print("✓ Sistema de verificación EPP iniciado")
    print("Presiona 'q' para salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección
        results = model(frame, verbose=False)

        # Analizar detecciones
        detecciones = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASSES[cls]
            if conf > 0.5:  # Solo confianza > 50%
                detecciones.append(name)

        # Verificar cumplimiento
        tiene_casco = 'Hardhat' in detecciones
        tiene_chaleco = 'Safety Vest' in detecciones
        tiene_mascara = 'Mask' in detecciones

        falta_casco = 'NO-Hardhat' in detecciones
        falta_chaleco = 'NO-Safety Vest' in detecciones
        falta_mascara = 'NO-Mask' in detecciones

        hay_persona = 'Person' in detecciones

        # Determinar estado
        if hay_persona:
            if falta_casco or falta_chaleco or falta_mascara:
                estado = "NO CUMPLE EPP"
                color = (0, 0, 255)  # Rojo
            elif tiene_casco and tiene_chaleco:
                estado = "CUMPLE EPP"
                color = (0, 255, 0)  # Verde
            else:
                estado = "VERIFICANDO..."
                color = (0, 255, 255)  # Amarillo
        else:
            estado = "SIN PERSONA"
            color = (128, 128, 128)  # Gris

        # Dibujar resultados
        annotated_frame = results[0].plot()

        # Agregar estado en pantalla
        cv2.rectangle(annotated_frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(annotated_frame, estado, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Mostrar EPP detectado
        y_pos = 130
        cv2.putText(annotated_frame, f"Casco: {'SI' if tiene_casco else 'NO'}",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if tiene_casco else (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Chaleco: {'SI' if tiene_chaleco else 'NO'}",
                    (20, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if tiene_chaleco else (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Mascara: {'SI' if tiene_mascara else 'NO'}",
                    (20, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if tiene_mascara else (0, 0, 255), 2)

        # Mostrar frame
        cv2.imshow("Verificacion EPP - Presiona 'q' para salir",
                   annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_compliance()
