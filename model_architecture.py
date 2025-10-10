# model_architecture.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications import DenseNet121

class TBModelBuilder:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_base_model(self, base_model_name='densenet121'):
        """Crea el modelo base con transfer learning"""
        
        # Estrategia para prevenir overfitting en dataset pequeño
        if base_model_name == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Modelo base no soportado: {base_model_name}")
        
        # Congelar capas base inicialmente
        base_model.trainable = False
        
        return base_model
    
    def build_model(self, base_model_name='densenet121', learning_rate=1e-4):
        """Construye el modelo completo"""
        
        # Modelo base
        base_model = self.create_base_model(base_model_name)
        
        # Capas personalizadas
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Capa base
        x = base_model(inputs, training=False)
        
        # Capas de clasificación
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Capas densas con regularización
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Capa de salida
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        # Modelo completo
        model = models.Model(inputs, outputs)
        
        # Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model, base_model
    
    def unfreeze_layers(self, model, base_model, unfreeze_ratio=0.5):
        """Descongela capas del modelo base para fine-tuning"""
        # Descongelar las últimas capas del modelo base
        base_model.trainable = True
        
        # Congelar primeras capas, descongelar últimas
        num_layers = len(base_model.layers)
        unfreeze_from = int(num_layers * (1 - unfreeze_ratio))
        
        for layer in base_model.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in base_model.layers[unfreeze_from:]:
            layer.trainable = True
        
        # Recompilar con learning rate más bajo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model