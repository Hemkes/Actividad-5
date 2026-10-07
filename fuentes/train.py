import mlflow
import os

# Configurar el directorio local donde se guardarán los registros (tracking URI)
# Esto creará automáticamente la carpeta 'mlruns' en tu directorio de trabajo
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Asignar un nombre al experimento
mlflow.set_experiment("Validacion_MLflow_Actividad5")

# Iniciar la ejecución (run) para registrar los datos
with mlflow.start_run(run_name="Prueba_Inicial"):
    print("Iniciando validación de MLflow...")
    
    # 1. Registrar Parámetros
    mlflow.log_param("algoritmo_prueba", "RandomForest")
    mlflow.log_param("n_estimadores", 100)
    
    # 2. Registrar Métricas
    mlflow.log_metric("exactitud_prueba", 0.95)
    mlflow.log_metric("error_prueba", 0.05)
    
    # 3. Registrar Artefactos (guardar un archivo físico)
    ruta_artefacto = "reporte_prueba.txt"
    with open(ruta_artefacto, "w") as f:
        f.write("Este es un archivo de texto generado durante la ejecución para validar los artefactos en MLflow.")
    
    mlflow.log_artifact(ruta_artefacto)
    
    print("¡Registro completado exitosamente!")