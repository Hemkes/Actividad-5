# Actividad 5: Entrenamiento, Ajuste y Registro de Modelos con MLflow

Este repositorio contiene la implementación de un flujo completo de Machine Learning utilizando scikit-learn y MLflow. Se compara el rendimiento de dos algoritmos de clasificación (Regresión Logística y Random Forest) sobre el conjunto de datos de Cáncer de Mama de scikit-learn.

## Estructura del Repositorio

- `datos/datos_ini/`: Contiene el dataset original en formato CSV (generado por el script).
- `fuentes/datos_prep.py`: Script para descargar y versionar los datos originales.
- `fuentes/train.py`: Script principal que ejecuta el entrenamiento, la validación cruzada (GridSearchCV) y el registro en MLflow.
- `evidencia_mlflow.csv` / `mlflow_ui.png`: Evidencias de los resultados exportados desde la interfaz de MLflow.

## Algoritmos Seleccionados
1. **Regresión Logística:** Evaluado con diferentes valores de regularización (`C`) y optimizadores (`solver`).
2. **Random Forest Classifier:** Evaluado modificando la cantidad de estimadores, profundidad máxima y división mínima de muestras.

Ambos modelos fueron ajustados mediante `GridSearchCV` con 5 k-folds, optimizando la métrica **F1-Score** para equilibrar falsos positivos y negativos en un contexto médico.

## Instrucciones de Reproducción

1. **Instalar dependencias:**
   ```bash
   pip install mlflow scikit-learn pandas