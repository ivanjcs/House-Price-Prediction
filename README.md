# 🏠 Ames Real Estate Pricing Engine | End-to-End ML Architecture

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-%23178C3A.svg?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-FF694B?logo=dbt&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-blue?logo=optuna&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-669DF6?logo=google-bigquery&logoColor=white)

## 📌 Visión General del Negocio

Este proyecto implementa un motor de predicción de precios inmobiliarios altamente optimizado. Superando el enfoque tradicional de cuadernos de análisis estáticos, el sistema está diseñado como una arquitectura de datos completa (**End-to-End**), abarcando desde la ingesta y transformación en Data Warehouse hasta el despliegue mediante una **API RESTful**.

> **🎯 Objetivo Central:** Proporcionar valoraciones precisas de mercado minimizando la varianza del error y controlando la asimetría extrema típica de los datos financieros.

### 📊 Métricas de Impacto

| Métrica | Resultado |
| :--- | :--- |
| **Kaggle Benchmark (RMSLE)** | `[0.12620]` |
| **Margen de Error Promedio** | `[12%]` <br> *(~ `$20,263` sobre una propiedad promedio)* |

---

## 🏗️ Arquitectura del Sistema Full-Stack

El proyecto adopta un enfoque modular que separa la ingeniería de datos del modelado predictivo, previniendo activamente el *Data Leakage*.

*   **🗄️ Capa de Datos (ELT):** [Repositorio de dbt aquí](#) *(<- Agrega el link a tu repo de dbt)*
    *   Procesamiento ejecutado en **Google BigQuery**.
    *   Limpieza, estandarización e imputación base gestionada con **dbt**.
*   **🧠 Capa de Machine Learning:**
    *   *Feature Engineering* dinámico orientado a negocio (ej. Ratios de rentabilidad, segmentación de amenidades).
    *   Algoritmo core: **XGBoost Regressor** con ajuste bayesiano (**Optuna**).
*   **🌐 Capa de Servicio:**
    *   Inferencia en tiempo real expuesta mediante **FastAPI**.
    *   Validación estricta de payloads con **Pydantic**.

---

## ⚙️ Hitos Técnicos y Rigor Analítico

Para asegurar la robustez del modelo en producción, se implementaron las siguientes estrategias avanzadas:

- **📈 Mitigación de Asimetría (*Skewness*):** Aplicación estructurada de transformaciones `np.log1p` sobre el vector objetivo y predictores de cola larga, recuperando resolución matemática en el algoritmo de agrupamiento de XGBoost (`tree_method='hist'`).
- **🗺️ Clustering Espacial:** Uso de `TargetKMeansClusterer` para reducir la alta cardinalidad de los vecindarios a 3 macro-zonas de valor económico, sin exponer promedios directos al modelo predictivo.
- **✂️ Selección de Características Determinista:** Poda automatizada de variables ruidosas mediante la extracción de *F-scores* (Frecuencia de división) del *Booster*, eliminando dimensiones con impacto nulo ($F \le 2$).
- **🎯 Optimización Bayesiana Segura:** Búsqueda de hiperparámetros con **Optuna** + *Early Stopping* sobre una matriz estrictamente podada, evaluada mediante un bucle de Validación Cruzada *Out-of-Fold* (OOF).

<!-- 💡 CONSEJO: Descomenta este bloque e inserta las rutas de tus imágenes (SHAP values, Optuna History) para darle un impacto visual tremendo al portfolio -->
<!-- 
<p align="center">
  <img src="ruta/a/tu/grafico_shap.png" width="45%" alt="Análisis de SHAP Values">
  <img src="ruta/a/tu/optuna_history.png" width="45%" alt="Historia de Optimización con Optuna">
</p>
-->

---

## 📂 Estructura del Proyecto

Adhiriendo a los estándares de *Cookiecutter Data Science*:

```text
ames_pricing_system/
├── api/                     # Capa de servicio RESTful
│   ├── main.py              # Endpoints de FastAPI
│   └── schemas.py           # Validaciones de entrada de datos (Pydantic)
│
├── notebooks/               # Experimentación y EDA
│   ├── 01_eda_and_drift_audit.ipynb 
│   └── 02_model_development.ipynb    
│
├── src/                     # Código fuente de producción (Pipeline OOP)
│   ├── features.py          # Clases personalizadas (UniversalFeatureEngineer)
│   ├── data_ingestion.py    # Conectores y splitters (Muro de Hierro de Validación)
│   └── modeling.py          # Pipeline maestro y evaluadores
│
├── models/                  # Artefactos serializados (.joblib)
├── requirements.txt         # Dependencias del entorno
└── README.md                # Documentación del sistema
```

# 🚀 Cómo ejecutar el proyecto
1. Entorno Virtual y Dependencias
Bash


# Crear y activar entorno virtual
(insertar poetry)


# Instalar dependencias
`pip install -r requirements.txt`
2. Entrenamiento del Modelo
Para replicar el preprocesamiento, ejecutar la poda por F-score, afinar hiperparámetros y generar el artefacto .joblib en la carpeta models/:


python src/train.py
3. Levantar la API de Predicción
```Bash
cd api
uvicorn main:app --reload
```

💡 Nota: La documentación interactiva de la API (Swagger UI) estará disponible automáticamente en http://localhost:8000/docs. Allí podrás probar predicciones en tiempo real.

Diseñado y desarrollado para demostrar rigor en analítica predictiva y arquitectura de datos.