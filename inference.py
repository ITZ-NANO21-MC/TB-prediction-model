# inference.py
import tensorflow as tf
import cv2
import numpy as np
import os
from typing import List, Dict, Union

class TBInference:
    def __init__(self, model_path: str, img_size: tuple = (224, 224)):
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.img_size = img_size
            self.class_names = ['Normal', 'Tuberculosis']
            
            # Verificar formato del modelo
            self._validate_model_output()
            
        except Exception as e:
            raise RuntimeError(f"Error cargando el modelo: {str(e)}")
    
    def _validate_model_output(self):
        """Valida que el modelo tenga el formato esperado"""
        # Crear input dummy para probar el modelo
        dummy_input = np.random.random((1, *self.img_size, 3)).astype(np.float32)
        dummy_pred = self.model.predict(dummy_input, verbose=0)
        
        # Verificar formato de salida
        if len(dummy_pred.shape) != 2 or dummy_pred.shape[1] != 1:
            print(f"⚠️ Advertencia: Formato de salida inesperado: {dummy_pred.shape}")
            print("El modelo podría necesitar ajustes en la lógica de predicción")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocesa una imagen para inferencia"""
        # Verificar que la imagen existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Verificar dimensiones
        if img.size == 0:
            raise ValueError(f"Imagen vacía o corrupta: {image_path}")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, self.img_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        
        # Añadir dimensión de batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _process_prediction(self, prediction: np.ndarray) -> Dict:
        """Procesa la predicción del modelo de forma robusta"""
        # Manejar diferentes formatos de salida
        if prediction.shape == (1, 1):  # Formato esperado
            confidence = float(prediction[0][0])
        elif prediction.shape == (1, 2):  # Probabilidades para ambas clases
            confidence = float(prediction[0][1])  # Asume tuberculosis es clase 1
        else:
            # Intentar extraer la probabilidad de forma genérica
            confidence = float(prediction.ravel()[0])
        
        # Determinar clase
        class_idx = 1 if confidence > 0.5 else 0
        class_name = self.class_names[class_idx]
        
        return {
            'class': class_name,
            'confidence': confidence,
            'class_index': class_idx
        }
    
    def predict(self, image_path: str) -> Dict:
        """Realiza predicción en una imagen"""
        try:
            # Preprocesar
            processed_img = self.preprocess_image(image_path)
            
            # Predecir
            prediction = self.model.predict(processed_img, verbose=0)
            
            # Procesar resultado
            result = self._process_prediction(prediction)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error en predicción para {image_path}: {str(e)}")
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Realiza predicción en múltiples imágenes"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                result['success'] = True
                results.append(result)
            except Exception as e:
                print(f"❌ Error procesando {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Ejemplo de uso mejorado
if __name__ == "__main__":
    try:
        # Inicializar inferencia
        inference = TBInference('models/saved_models/tb_final_model.h5') # Ruta del modelo
        
        # Ejemplo de predicción única
        result = inference.predict('img0.png') # Ruta de la imagen
        print(f"🔍 Resultado: {result['class']} (Confianza: {result['confidence']:.3f})")
        
        # Ejemplo de predicción múltiple
        image_paths = ['img1.png', 'img2.png', 'img3.png']
        results = inference.predict_batch(image_paths)
        
        for res in results:
            if res['success']:
                print(f"✅ {res['image_path']}: {res['class']} ({res['confidence']:.3f})")
            else:
                print(f"❌ {res['image_path']}: Error - {res['error']}")
                
    except Exception as e:
        print(f"💥 Error inicializando el sistema: {str(e)}")