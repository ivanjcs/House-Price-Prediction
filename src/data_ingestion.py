import pandas as pd
from google.cloud import bigquery

# Importamos config para que al ejecutarse este script, 
# se carguen las variables de entorno automáticamente.
import config 

def get_bq_client() -> bigquery.Client:
    """Inicializa y retorna el cliente de BigQuery de forma segura."""
    # GCP busca automáticamente la variable GOOGLE_APPLICATION_CREDENTIALS 
    # que ya fue cargada en el entorno por config.py
    try:
        client = bigquery.Client()
        print("✅ Cliente de BigQuery inicializado correctamente.")
        return client
    except Exception as e:
        print(f"❌ Error al conectar con BigQuery: {e}")
        raise

def get_train_data(client: bigquery.Client) -> pd.DataFrame:
    """Descarga los datos de entrenamiento desde la capa Gold de dbt."""
    query = """
        SELECT *
        FROM `primer-proyecto-kaggle.dbt_icastro_gold_marts.obt_house_prices__train`
    """
    print("⏳ Descargando datos de entrenamiento (Train)...")
    df_train = client.query(query).to_dataframe()
    
    # Limpieza básica de la ingesta
    if 'property_id' in df_train.columns:
        df_train = df_train.drop(columns=['property_id'])
        
    print(f"✅ Datos de Train listos. Filas: {df_train.shape[0]}, Columnas: {df_train.shape[1]}")
    return df_train

def get_test_data(client: bigquery.Client) -> tuple[pd.DataFrame, pd.Series]:
    """Descarga los datos de testeo y separa los IDs para la predicción final."""
    query_test = """
        SELECT *
        FROM `primer-proyecto-kaggle.dbt_icastro_gold_marts.obt_house_prices__test`
    """
    print("🔍 Descargando datos de test (Kaggle Test)...")
    df_test = client.query(query_test).to_dataframe()
    
    # Validación de datos y auditoría básica
    duplicados = df_test[df_test.duplicated(subset=['property_id'], keep=False)]
    if not duplicados.empty:
        print("🚨 ADVERTENCIA: Se encontraron casas duplicadas en Test.")
        print(duplicados[['property_id', 'Neighborhood', 'SaleCondition']])

    # Separar IDs y limpiar
    ids_submission = df_test['property_id']
    X_test = df_test.drop(columns=['property_id', 'Id'], errors='ignore')
    
    print(f"✅ Datos de Test listos. Filas: {X_test.shape[0]}")
    return X_test, ids_submission

# Este bloque permite probar el script individualmente 
if __name__ == "__main__":
    bq_client = get_bq_client()
    df_entrenamiento = get_train_data(bq_client)
    df_prueba, ids_prueba = get_test_data(bq_client)