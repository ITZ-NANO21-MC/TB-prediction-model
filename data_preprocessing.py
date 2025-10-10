# data_preprocessing
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import albumentations as A

class TBDataPreprocessor:
    def __init__(self, data_path, img_size=(224, 224)):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = ['Normal', 'Tuberculosis']
        self.setup_data_augmentation()
    
    def setup_data_augmentation(self):
        """Configura las transformaciones para data augmentation"""
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                             rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, 
                          fill_value=0, p=0.2),
        ])
    
    def load_and_preprocess_images(self):
        """Carga y preprocesa todas las imágenes del dataset"""
        images = []
        labels = []
        
        print("🔍 Cargando imágenes del dataset...")
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Advertencia: No se encontró la carpeta {class_path}")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Cargar y preprocesar imagen
                    img = self.load_single_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_idx)
        
        print(f"✅ Imágenes cargadas: {len(images)}")
        print(f"📊 Distribución: Normal={labels.count(0)}, TB={labels.count(1)}")
        
        return np.array(images), np.array(labels)
    
    def load_single_image(self, img_path):
        """Carga y preprocesa una sola imagen"""
        try:
            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Error leyendo: {img_path}")
                return None
            
            # Convertir BGR a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            img = cv2.resize(img, self.img_size)
            
            # Normalizar a [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"❌ Error procesando {img_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image, label):
        """Aplica data augmentation a una imagen"""
        augmented = self.train_transform(image=image)
        return augmented['image'], label
    
    def create_tf_datasets(self, validation_split=0.2, batch_size=8):
        """Crea datasets de TensorFlow para entrenamiento y validación"""
        
        # Cargar datos
        images, labels = self.load_and_preprocess_images()
        
        if len(images) == 0:
            raise ValueError("❌ No se encontraron imágenes para procesar")
        
        # Dividir en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, 
            stratify=labels, random_state=42
        )
        
        print(f"📈 Conjunto de entrenamiento: {len(X_train)} imágenes")
        print(f"📊 Conjunto de validación: {len(X_val)} imágenes")
        
        # Crear datasets de TensorFlow
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        # Aplicar aumentación solo al conjunto de entrenamiento
        def augment_fn(image, label):
            image_aug = tf.numpy_function(
                func=lambda x: self.train_transform(image=x)['image'],
                inp=[image],
                Tout=tf.float32
            )
            image_aug.set_shape(image.shape)
            return image_aug, label
        
        train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Configurar datasets para mejor performance
        train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, (X_train, X_val, y_train, y_val)