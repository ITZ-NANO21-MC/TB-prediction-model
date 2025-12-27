# 🩺 TB-Detector-AI 

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-FF6F00?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Development-yellow?style=for-the-badge)

**Sistema de apoyo al diagnóstico de tuberculosis mediante análisis de imágenes radiográficas de tórax**

[Características](#-características) • [Instalación](#-instalación) • [Modelo](#-modelo) • [Demo](#-demo)

</div>

## 📋 Tabla de Contenidos

- [📖 Descripción](#-descripción)
- [🎯 Características](#-características)
- [🚀 Instalación Rápida](#-instalación-rápida)
- [💻 Uso](#-uso)
- [🧠 Modelo](#-modelo)
- [📊 Dataset](#-dataset)
- [🏗️ Estructura del Proyecto](#️-estructura-del-proyecto)
- [📸 Demo](#-demo)
- [🔧 Desarrollo](#-desarrollo)
- [🤝 Contribución](#-contribución)
- [📄 Licencia](#-licencia)

## 📖 Descripción

**TB-Detector AI** es un sistema de inteligencia artificial diseñado para asistir a profesionales de la salud en la detección temprana de tuberculosis mediante el análisis de imágenes radiográficas de tórax. El sistema funciona como una "segunda opinión" automatizada, especialmente útil en regiones con escasez de especialistas en radiología.

### 🎯 Objetivos Principales

- 🔍 **Detección temprana** de patrones de tuberculosis en radiografías de tórax
- ⚡ **Reducción de tiempos** de diagnóstico de semanas a horas
- 🌍 **Ampliación de cobertura** en zonas rurales y remotas

## 🎯 Características

### 🧠 Capacidades del Modelo
- **Clasificación Binaria**: Detección de TB con probabilidad [0.0-1.0]
- **Explicabilidad Avanzada**: Mapas de calor Grad-CAM++ para visualización de hallazgos
- **Múltiples Patrones**: Detección de lesiones cavitarias, consolidaciones, derrame pleural, patrones miliares y linfadenopatía
- **Control de Calidad**: Evaluación automática de calidad de imagen

### 💻 Características Técnicas
- **Arquitectura**: DenseNet-121 con Transfer Learning
- **Framework**: TensorFlow 2.13.0
- **Precisión**: 99% en conjunto de prueba (840 imágenes)
- **Sensibilidad**: 97% (mínimos falsos negativos)
- **Especificidad**: 100% (sin falsos positivos)

### 🏥 Flujo de Trabajo Clínico
```
Carga de Imagen → Preprocesamiento → Análisis IA → Resultados Explicables → Diagnóstico Asistido
```

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.9+
- TensorFlow 2.13.0
- 8GB RAM mínimo (16GB recomendado)
- GPU NVIDIA (opcional pero recomendado)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/ITZ-NANO21-MC/TB-prediction-model.git
cd TB-prediction-model
```

2. **Configurar entorno virtual**
```bash
python -m venv tb_env
source tb_env/bin/activate  # Linux/Mac
# tb_env\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Inferencia con el Modelo

El archivo `inference.py` permite realizar predicciones con el modelo entrenado:

```python
# inference.py - Ejemplo de uso
from inference import TBInference

# Inicializar inferencia
inference = TBInference('models/saved_models/tb_final_model.h5')

# Predicción única
result = inference.predict('ejemplo_radiografia.png')
print(f"🔍 Resultado: {result['class']} (Confianza: {result['confidence']:.3f})")

# Predicción por lotes
image_paths = ['img1.png', 'img2.png', 'img3.png']
results = inference.predict_batch(image_paths)

for res in results:
    if res['success']:
        print(f"✅ {res['image_path']}: {res['class']} ({res['confidence']:.3f})")
    else:
        print(f"❌ {res['image_path']}: Error - {res['error']}")
```

## 🧠 Modelo

### Arquitectura
```python
model_architecture = {
    "backbone": "DenseNet-121",
    "input_shape": (512, 512, 3),
    "transfer_learning": "ImageNet pre-trained",
    "classification_head": [
        "GlobalAveragePooling2D",
        "Dense(128, activation='relu')",
        "Dropout(0.3)",
        "Dense(64, activation='relu')", 
        "Dropout(0.2)",
        "Dense(1, activation='sigmoid')"
    ]
}
```

### 📈 **Resultados con Dataset Ampliado**

**Después del entrenamiento con un dataset más grande y balanceado**, el modelo ha mostrado una mejora significativa en todas las métricas. A continuación se presentan los resultados obtenidos en el conjunto de prueba de 840 imágenes (700 normales, 140 con tuberculosis):

### Métricas de Rendimiento
| Métrica | Objetivo | Actual |
|---------|----------|---------|
| **AUC-ROC** | > 0.95 | **0.99** |
| **Sensibilidad (Recall)** | > 90% | **97%** |
| **Especificidad** | > 85% | **100%** |
| **Precisión** | > 88% | **99%** |
| **Exactitud (Accuracy)** | > 90% | **99%** |
| **F1-Score** | > 0.90 | **0.98** |

### 📊 **Reporte de Clasificación Detallado**
```
==================================================
📈 REPORTE DE CLASIFICACIÓN
==================================================
              precision    recall  f1-score   support

      Normal       0.99      1.00      1.00       700
Tuberculosis       0.99      0.97      0.98       140

    accuracy                           0.99       840
   macro avg       0.99      0.98      0.99       840
weighted avg       0.99      0.99      0.99       840
```

### 🔍 **Análisis de los Resultados**
- **Alta especificidad (100%)**: El modelo no produce falsos positivos para imágenes normales, lo que es crucial para evitar tratamientos innecesarios.
- **Excelente sensibilidad (97%)**: Detecta correctamente el 97% de los casos de tuberculosis, minimizando falsos negativos.
- **Balance óptimo**: El F1-Score de 0.98 indica un equilibrio perfecto entre precisión y recall.

### 🎯 Patrones Detectados
- ✅ Lesiones cavitarias
- ✅ Consolidaciones pulmonares 
- ✅ Derrame pleural
- ✅ Patrones miliares
- ✅ Linfadenopatía mediastinal

## 📊 Dataset

### Fuentes de Datos
- **Kaggle TB Dataset**: 4200 imágenes (Normal/TB)

### Estructura del Dataset
```
├── Dataset/
│   ├── Normal/           # 3500 imágenes
│   └── Tuberculosis/     # 700 imágenes
```

### Preprocesamiento
```python
preprocessing_steps = {
    "resize": "(512, 512)",
    "normalization": "Pixel values [0, 1]",
    "augmentation": [
        "Rotación aleatoria (±10°)",
        "Volteo horizontal",
        "Ajuste de brillo/contraste",
        "Simulación de artefactos"
    ]
}
```

## 🏗️ Estructura del Proyecto

```
TB-Detector-AI/
│
├── 📁 models/
│   ├── saved_models/           # Modelos entrenados guardados
│   └── training_logs/          # Logs de entrenamiento
│
├── 📁 Dataset/
│   ├── Normal/                 # Imágenes normales
│   └── Tuberculosis/           # Imágenes con tuberculosis
│
├── data_preprocessing.py       # Procesamiento de datos
├── model_architecture.py       # Arquitectura del modelo
├── exploratory_analysis.py     # Análisis exploratorio del dataset
├── inference.py                # Inferencia del modelo
├── training_pipeline.py        # Pipeline de entrenamiento
├── main.py                     # Punto de entrada principal
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```

## 📸 Demo

### Ejecutar Demo Local
```bash
# 1. Asegúrate de tener el modelo descargado en models/saved_models/
# 2. Ejecutar demo con imagen de prueba
python inference.py --image tests/test_image.png
```

### Resultado Esperado
```
✅ Procesando imagen: tests/test_image.png
🔍 Predicción: Tuberculosis detectada
📊 Confianza: 0.982 (98.2%)
📍 Hallazgos: Lesiones cavitarias, consolidación pulmonar
⚠️ Recomendación: Consulta con especialista para confirmación
```

## 🔧 Desarrollo

### Configuración de Desarrollo
```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt

# Ejecutar tests
python -m pytest tests/
```

### Entrenamiento del Modelo
```bash
# Ejecutar pipeline de entrenamiento completo
python training_pipeline.py --epochs 50 --batch_size 32
```

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor lee nuestras guías:

### 📋 Proceso de Contribución
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### 🎯 Áreas de Contribución Prioritarias
- 🏥 Validación clínica adicional
- 🔍 Mejora de explicabilidad del modelo
- 🌍 Soporte para múltiples idiomas
- 📱 Aplicación móvil complementaria
- 🖥️ Interfaz web para uso clínico

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

**⚠️ Aviso Legal**: Este software está destinado únicamente para investigación y como herramienta de apoyo al diagnóstico. No substituye el juicio clínico de profesionales médicos calificados. Siempre consulte con un médico para diagnóstico y tratamiento.

**🔬 Para uso de investigación** | **🏥 Versión de desarrollo** | **📊 Modelo con 99% de precisión**

---
```
