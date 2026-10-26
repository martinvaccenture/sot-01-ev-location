import logging
import os
import sys
from datetime import datetime

from bronze.ingesta_puntos_recarga import descargar_dgt as descargar_dgt_puntos
from bronze.ingestion_municipal import *
from silver.silver_municipal import *
from silver.silver_cargadores_geometria import *
from gold.gold import *

# ============================================================
# RUTA RAÍZ DEL PROYECTO
# ============================================================

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))
    
from config_modelo import (
    NORMALIZACION,
    PESOS_MODELO,
    ESCENARIOS,
    VALORES_RIESGO_EXCLUIDOS,
    UMBRALES_RIESGO_FLOTA
)

from dotenv import load_dotenv


load_dotenv()

# ============================================================
# VARIABLES DE ENTORNO
# ============================================================

AWS_REGION = os.getenv("AWS_REGION")
RAW_PATH = os.getenv("S3_RAW_PATH")
SILVER_PATH = os.getenv("S3_SILVER_PATH")
GOLD_PATH = os.getenv("S3_GOLD_PATH")


POTENCIA_MINIMA = 50  # kW


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# GUARDAR TABLAS RESULTADOS
# ============================================================

def guardar_tablas_resultados(preparacion):

    for nombre, dataframe in preparacion["tablas_escenarios"].items():

        ruta = f"{GOLD_PATH}{nombre}.parquet"

        dataframe.to_parquet(
            ruta,
            index=False
        )

        logger.info(
            "Saved scenario table: %s",
            ruta
        )


    for nombre, dataframe in preparacion["tablas_top"].items():

        ruta = f"{GOLD_PATH}{nombre}.parquet"

        dataframe.to_parquet(
            ruta,
            index=False
        )

        logger.info(
            "Saved final ranking table: %s",
            ruta
        )


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_pipeline():

    start_time = datetime.now()

    logger.info("==========================================")
    logger.info("Starting EV Location Intelligence pipeline")
    logger.info("==========================================")

    try:

        # ----------------------------------------------------
        # 1. INGESTION
        # ----------------------------------------------------

        logger.info("Starting ingestion puntos recarga...")

        descargar_dgt_puntos()

        logger.info("Ingestion puntos recarga completed successfully.")

        logger.info("Starting ingestion municipal...")

        descargar_dgt()
        descargar_padron()
        descargar_eoh()

        logger.info("Ingestion municipal completed successfully.")


        # ----------------------------------------------------
        # 2. SILVER
        # ----------------------------------------------------

        logger.info("Starting Silver layer municipal...")

        construir_municipios()
        construir_vehiculos()

        logger.info("Silver layer municipal completed successfully.")

        logger.info("Starting Silver layer cargadores_geometria...")

        df = leer_puntos_recarga_xml(
            f"{RAW_PATH}dgt/electrolineras.xml"
        )

        gdf_recarga = convertir_coordenadas_en_geometrias(df)

        gdf_filtrado = filtrar_recarga_minima(
            gdf_recarga,
            POTENCIA_MINIMA
        )

        gdf_municipios = preparar_municipios()

        gdf_join = realizar_spatial_join(
            gdf_filtrado,
            gdf_municipios
        )

        metricas, gdf_cruces_fallidos = medir_cruces_fallidos(
            gdf_join
        )

        gdf_final = agregar_recarga_por_municipio(
            gdf_join,
            gdf_municipios
        )

        guardar_resultados(
            gdf_final,
            gdf_cruces_fallidos
        )

        logger.info("Silver layer cargadores_geometria completed successfully.")


        # ----------------------------------------------------
        # 3. GOLD
        # ----------------------------------------------------

        logger.info("Starting Gold layer...")

        gold = generar_gold(
            SILVER_PATH,
            f"{GOLD_PATH}gold.parquet",
            PESOS_MODELO,
            NORMALIZACION
        )

        modelo, baseline = comparar_con_baseline(
            gold
        )

        resultados_kmeans, metricas = analizar_kmeans(
            gold
        )

        preparacion = preparar_escenarios(
            gold,
            ESCENARIOS,
            NORMALIZACION,
            UMBRALES_RIESGO_FLOTA,
            VALORES_RIESGO_EXCLUIDOS
        )

        guardar_tablas_resultados(
            preparacion
        )

        logger.info("Gold layer completed successfully.")


        # ----------------------------------------------------
        # PIPELINE COMPLETED
        # ----------------------------------------------------

        execution_time = datetime.now() - start_time

        logger.info("==========================================")
        logger.info("Pipeline completed successfully.")
        logger.info(f"Execution time: {execution_time}")
        logger.info("==========================================")


    except Exception as error:

        logger.exception(
            "Pipeline failed: %s",
            error
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()