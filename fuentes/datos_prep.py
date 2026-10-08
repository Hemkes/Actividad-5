import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def descargar_y_preparar_datos():
    """Descarga el dataset de Breast Cancer y lo guarda en la carpeta de datos."""
    print("Descargando dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # 1. Obtiene la ruta absoluta de la carpeta donde está este script
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Sube un nivel para llegar a la carpeta principal ('Actividad-5')
    directorio_raiz = os.path.dirname(directorio_script)
    
    # 3. Construye la ruta hacia 'datos/datos_ini' desde la raíz
    directorio_salida = os.path.join(directorio_raiz, 'datos', 'datos_ini')
    
    # Asegurar que el directorio de salida existe
    os.makedirs(directorio_salida, exist_ok=True)
    
    # 4. Define la ruta final del archivo CSV
    ruta_salida = os.path.join(directorio_salida, 'breast_cancer_dataset.csv')
    
    df.to_csv(ruta_salida, index=False)
    print(f"Datos guardados exitosamente en: {ruta_salida}")

if __name__ == "__main__":
    descargar_y_preparar_datos()