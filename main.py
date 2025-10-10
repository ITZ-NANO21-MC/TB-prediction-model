# main.py
import tensorflow as tf
from data_preprocessing import TBDataPreprocessor
from model_architecture import TBModelBuilder
from training_pipeline import TBTrainingPipeline
import os

def main():
    print("🩺 INICIANDO ENTRENAMIENTO DEL MODELO DE DETECCIÓN DE TB")
    print("=" * 60)
    
    # Configuración
    DATA_PATH = "/content/proyecto/Dataset"  # Ajusta esta ruta según tu estructura
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS_PHASE1 = 30
    EPOCHS_PHASE2 = 20
    
    # Verificar que existe el dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: No se encuentra el directorio {DATA_PATH}")
        print("💡 Asegúrate de que la estructura sea: Dataset/Normal/ y Dataset/Tuberculosis/")
        return
    
    # 1. Preprocesamiento de datos
    print("📁 Paso 1: Preprocesando datos...")
    preprocessor = TBDataPreprocessor(DATA_PATH, IMG_SIZE)
    train_dataset, val_dataset, (X_train, X_val, y_train, y_val) = \
        preprocessor.create_tf_datasets(batch_size=BATCH_SIZE)
    
    # 2. Construcción del modelo
    print("🧠 Paso 2: Construyendo modelo...")
    model_builder = TBModelBuilder(input_shape=(*IMG_SIZE, 3))
    model, base_model = model_builder.build_model(learning_rate=1e-4)
    
    print("📋 Resumen del modelo:")
    model.summary()
    
    # 3. Fase 1: Entrenamiento con capas base congeladas
    print("🎯 Paso 3: Fase 1 - Entrenamiento inicial...")
    training_pipeline = TBTrainingPipeline(model, 'tb_phase1')
    history_phase1 = training_pipeline.train_model(
        train_dataset, val_dataset, epochs=EPOCHS_PHASE1
    )
    
    # 4. Fase 2: Fine-tuning con capas descongeladas
    print("🔧 Paso 4: Fase 2 - Fine-tuning...")
    model = model_builder.unfreeze_layers(model, base_model, unfreeze_ratio=0.3)
    
    training_pipeline_phase2 = TBTrainingPipeline(model, 'tb_phase2')
    history_phase2 = training_pipeline_phase2.train_model(
        train_dataset, val_dataset, epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1
    )
    
    # 5. Evaluación final
    print("📊 Paso 5: Evaluación final...")
    training_pipeline_phase2.evaluate_model(val_dataset, (X_val, y_val))
    
    # 6. Guardar modelo final
    print("💾 Paso 6: Guardando modelo final...")
    model.save(f'models/saved_models/tb_final_model.h5')
    print("✅ Modelo guardado en: models/saved_models/tb_final_model.h5")
    
    print("🎉 ¡Entrenamiento completado!")

if __name__ == "__main__":
    # Configurar GPU si está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✅ GPU detectada: {physical_devices[0]}")
    else:
        print("⚠️  No se detectó GPU, usando CPU")
    
    main()