import os
import mlflow

# 1. Subir un nivel desde 'fuentes/' a la carpeta raíz 'Actividad-5'
DIRECTORIO_FUENTES = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.dirname(DIRECTORIO_FUENTES) # Raíz del proyecto

RUTA_DB = os.path.join(DIRECTORIO_RAIZ, "mlflow.db")
RUTA_ARTEFACTOS = os.path.join(DIRECTORIO_RAIZ, "mlruns")

# 2. Configurar SQLite como el tracking URI
mlflow.set_tracking_uri(f"sqlite:///{RUTA_DB}")

# 3. Crear o seleccionar el experimento definiendo la ruta de artefactos en la raíz
NOMBRE_EXPERIMENTO = "Validacion_MLflow_Actividad5"
experimento = mlflow.get_experiment_by_name(NOMBRE_EXPERIMENTO)

if experimento is None:
    mlflow.create_experiment(
        name=NOMBRE_EXPERIMENTO,
        artifact_location=f"file:///{RUTA_ARTEFACTOS}"
    )

mlflow.set_experiment(NOMBRE_EXPERIMENTO)

# 4. Iniciar la ejecución
with mlflow.start_run(run_name="Prueba_Artefacto_2"):
    print("Iniciando validación de MLflow...")
    
    # Registrar Parámetros
    mlflow.log_param("algoritmo_prueba", "RandomForest")
    mlflow.log_param("n_estimadores", 100)
    
    # Registrar Métricas
    mlflow.log_metric("exactitud_prueba", 0.95)
    mlflow.log_metric("error_prueba", 0.05)
    
    # Registrar Artefactos
    ruta_artefacto = os.path.join(DIRECTORIO_RAIZ, "reporte_prueba.txt")
    with open(ruta_artefacto, "w", encoding="utf-8") as f:
        f.write("Este es un archivo de texto generado durante la ejecución para validar los artefactos en MLflow.")
    
    mlflow.log_artifact(ruta_artefacto)
    
    print("¡Registro completado exitosamente!")