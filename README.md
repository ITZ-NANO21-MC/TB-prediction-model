# 🩺 TB-Detector-AI 

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-FF6F00?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Development-yellow?style=for-the-badge)

**Sistema de apoyo al diagnóstico de tuberculosis mediante análisis de imágenes radiográficas de tórax**

[Características](#-características) • [Instalación](#-instalación) • [Modelo](#-modelo) 

</div>

## 📋 Tabla de Contenidos

- [📖 Descripción](#-descripción)
- [🎯 Características](#-características)
- [🚀 Instalación](#-instalación)
- [💻 Uso](#-uso)
- [🧠 Modelo](#-modelo)
- [📊 Dataset](#-dataset)
- [🏗️ Estructura del Proyecto](#️-estructura-del-proyecto)
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

### 🏥 Flujo de Trabajo Clínico
```
Carga de Imagen → Preprocesamiento → Análisis IA → Resultados Explicables → Diagnóstico Asistido
```

## 🚀 Instalación

### Prerrequisitos
- Python 3.9+
- TensorFlow 2.13.0
- 8GB RAM mínimo (16GB recomendado)
- GPU NVIDIA (opcional pero recomendado)

### Instalación Rápida

1. **Clonar el repositorio**
```bash
git clone https://github.com/ITZ-NANO21-MC/TB-prediction-model.git
cd TB-Detector-AI
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

4. 📱 **Probar el modelo**

# Modulo inference.py

```python
     # Inicializar inferencia
    inference = TBInference('models/saved_models/tb_final_model.h5') # Ruta del modelo
        
    # Ejemplo de predicción única
    result = inference.predict('img0.png') #ruta de la imagen
    print(f"🔍 Resultado: {result['class']} (Confianza: {result['confidence']:.3f})")

    # Ejemplo de predicción múltiple
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

### Métricas de Rendimiento
| Métrica | Objetivo | Actual |
|---------|----------|---------|
| **AUC-ROC** | > 0.95 | 0.69 |
| **Sensibilidad** | > 90% | 68% |
| **Especificidad** | > 85% | 70% |
| **Precisión** | > 88% | 67% |

### 🎯 Patrones Detectados
- ✅ Lesiones cavitarias
- ✅ Consolidaciones pulmonares  
- ✅ Derrame pleural
- ✅ Patrones miliares
- ✅ Linfadenopatía mediastinal

## 📊 Dataset

### Fuentes de Datos
- **Kaggle TB Dataset**: 3,500 imágenes (Normal/TB)
- **Datos usados para el entrenamiento**: Se utilizo un subconjunto del dataset de Kaggle TB.

### Estructura del Dataset
```
├── Dataset/
│   ├── Normal/           # 80 imágenes
│   └── Tuberculosis/     # 60 imágenes
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
│   ├── data_preprocessing.py  # Procesamiento de datos
│   ├── model_architecture.py  # Arquitectura del modelo
│   ├── exploratory_analysis.py # Análisis exploratorio del dataset
│   ├── inference.py           # Inferencia del modelo
│   ├── main.py                # Archiivo principal        
├── 📁 models/              
│   ├── saved_models/           # Modelos guardados
│   └── training_logs/
├── 📁 Dataset/              
│   ├── Normal/          
│   └── Tuberculosis/
├── requirements.txt
└── README.md
```

## 🔧 Desarrollo

### Configuración de Desarrollo
```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt

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

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

**Aviso Legal**: Este software está destinado únicamente para investigación y como herramienta de apoyo al diagnóstico. No substituye el juicio clínico de profesionales médicos calificados.

