# exploratory_analysis.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import TBDataPreprocessor

def exploratory_analysis():
    """Análisis exploratorio del dataset"""
    
    preprocessor = TBDataPreprocessor("Dataset")
    images, labels = preprocessor.load_and_preprocess_images()
    
    print("📊 ANÁLISIS EXPLORATORIO DEL DATASET")
    print("=" * 50)
    
    # Estadísticas básicas
    print(f"📁 Total de imágenes: {len(images)}")
    print(f"🩺 Casos Normales: {list(labels).count(0)}")
    print(f"🦠 Casos Tuberculosis: {list(labels).count(1)}")
    
    # Visualizar algunas imágenes
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    normal_count = 0
    tb_count = 0
    
    for i in range(8):
        if normal_count < 4 and labels[i] == 0:
            axes[i].imshow(images[i])
            axes[i].set_title('Normal')
            axes[i].axis('off')
            normal_count += 1
        elif tb_count < 4 and labels[i] == 1:
            axes[i].imshow(images[i])
            axes[i].set_title('Tuberculosis')
            axes[i].axis('off')
            tb_count += 1
    
    plt.suptitle('Muestras del Dataset - Normal vs Tuberculosis', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    exploratory_analysis()