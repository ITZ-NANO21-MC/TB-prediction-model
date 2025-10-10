# training_pipeline.py
import tensorflow as tf
import os
import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)

class TBTrainingPipeline:
    def __init__(self, model, model_name='tb_detector'):
        self.model = model
        self.model_name = model_name
        self.callbacks = []
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Configura los callbacks para el entrenamiento"""
        
        # Directorio para guardar modelos
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('models/training_logs', exist_ok=True)
        
        # Callback para guardar el mejor modelo
        checkpoint_cb = ModelCheckpoint(
            filepath=f'models/saved_models/{self.model_name}_best.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        # Early stopping para prevenir overfitting
        early_stopping_cb = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reducción de learning rate
        reduce_lr_cb = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        
        # TensorBoard para monitoreo
        log_dir = f"models/training_logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_cb = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
        
        self.callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]
    
    def train_model(self, train_dataset, val_dataset, epochs=50, initial_epoch=0):
        """Entrena el modelo"""
        
        print("🚀 Iniciando entrenamiento del modelo...")
        
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=val_dataset,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, val_dataset, val_data):
        """Evalúa el modelo en el conjunto de validación"""
        print("📊 Evaluando modelo...")
        
        # Evaluación estándar
        evaluation = self.model.evaluate(val_dataset, verbose=0)
        
        # Métricas adicionales
        X_val, y_val = val_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Reporte de clasificación
        print("\n" + "="*50)
        print("📈 REPORTE DE CLASIFICACIÓN")
        print("="*50)
        print(classification_report(y_val, y_pred_binary, 
                                  target_names=['Normal', 'Tuberculosis']))
        
        # Matriz de confusión
        cm = confusion_matrix(y_val, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Tuberculosis'],
                   yticklabels=['Normal', 'Tuberculosis'])
        plt.title('Matriz de Confusión - Detección de TB')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return evaluation