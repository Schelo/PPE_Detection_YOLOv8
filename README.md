# PPE Detection YOLOv8

Sistema de detección de Equipos de Protección Personal (EPP) en tiempo real utilizando YOLOv8.

## Descripción

Este proyecto implementa un sistema de visión por computador capaz de detectar si los trabajadores de construcción están utilizando correctamente su equipo de protección personal (EPP).

Desarrollado como proyecto académico para el curso de Inteligencia Artificial en INACAP.

## Clases Detectadas

| Clase | Descripción |
|-------|-------------|
| Hardhat | Casco de seguridad |
| Safety Vest | Chaleco reflectante |
| Mask | Mascarilla/Respirador |
| NO-Hardhat | Persona sin casco |
| NO-Safety Vest | Persona sin chaleco |
| NO-Mask | Persona sin mascarilla |
| Person | Persona |
| Safety Cone | Cono de seguridad |
| machinery | Maquinaria |
| vehicle | Vehículo |

## Requisitos

- Python 3.10+
- GPU con CUDA (recomendado, pero funciona con CPU)
- Webcam (para detección en tiempo real)

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/PPE_Detection_YOLOv8.git
cd PPE_Detection_YOLOv8
```

### 2. Crear entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install ultralytics opencv-python
```

## Uso

### Detectar en imagen
Analiza una imagen y muestra los EPP detectados.
```bash
python detect_image.py "ruta/a/imagen.jpg"
```

**Ejemplo:**
```bash
python detect_image.py "archive/demo_media/construction-safety.jpg"
```

### Detectar en video
Procesa un video y genera un archivo con las detecciones.
```bash
python detect_video.py "ruta/a/video.mp4"
```

**Ejemplo:**
```bash
python detect_video.py "archive/demo_media/hardhat.mp4"
```

### Detectar con cámara (tiempo real)
Abre la webcam y detecta EPP en tiempo real. Presiona 'q' para salir.
```bash
python detect_camera.py
```

### Verificar cumplimiento EPP
Detecta en tiempo real y muestra si la persona cumple con el EPP requerido (casco, chaleco, mascarilla).
```bash
python check_compliance.py
```

### Entrenar modelo (opcional)
Re-entrena el modelo con el dataset. Solo necesario si modificas el dataset.
```bash
python train.py
```

## Estructura del Proyecto
```text
PPE_Detection_YOLOv8/
├── archive/
│   ├── css-data/           # Dataset (train/valid/test)
│   └── demo_media/         # Imágenes y videos para pruebas
├── runs/
│   └── detect/
│       └── ppe_detector/   # Modelo entrenado
│           └── weights/
│               └── best.pt
├── check_compliance.py     # Verificación de cumplimiento EPP
├── data.yaml               # Configuración del dataset
├── detect_camera.py        # Detección con webcam
├── detect_image.py         # Detección en imágenes
├── detect_video.py         # Detección en videos
├── train.py                # Script de entrenamiento
├── yolov8n.pt              # Modelo base YOLOv8 nano
└── README.md
```

## Dataset

Dataset utilizado: [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

- 2801 imágenes base + 20 imágenes adicionales de EPP
- 10 clases
- Formato YOLOv8

## Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| mAP@50 | 73.0% |
| Precisión | 83.7% |
| Recall | 66.0% |

### Detección por clase

| Clase | Precisión | mAP50 |
|-------|-----------|-------|
| Hardhat | 97.4% | 82.3% |
| Mask | 93.7% | 82.2% |
| Safety Vest | 85.1% | 79.0% |
| Person | 81.1% | 77.8% |

## Autores

- Marcelo Hernández
- Karen [Apellido]

## Licencia

Este proyecto es para fines educativos - INACAP 2026.
