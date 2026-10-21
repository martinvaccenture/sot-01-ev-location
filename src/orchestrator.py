import logging
import sys
from datetime import datetime

from ingesta_puntos_recarga import descargar_dgt as descargar_dgt_puntos
from ingestion_municipal import *
from silver_municipal import *
from silver_cargadores_geometria import *
from gold import *
from dotenv import load_dotenv


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
RAW_PATH = os.getenv("S3_RAW_PATH")
SILVER_PATH = os.getenv("S3_SILVER_PATH")
GOLD_PATH = os.getenv("S3_GOLD_PATH")

PESOS_MODELO = [1 / 6, 1 / 3, 1 / 3, 1 / 6]


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

        df = leer_puntos_recarga_xml(f"{RAW_PATH}dgt/electrolineras.xml")
        gdf_recarga = convertir_coordenadas_en_geometrias(df)
        gdf_municipios = preparar_municipios()
        gdf_join = realizar_spatial_join(gdf_recarga, gdf_municipios)
        metricas = medir_cruces_fallidos(gdf_join)
        gdf_filtrado = filtrar_recarga_minima(gdf_join, POTENCIA_MINIMA)
        gdf_final = agregar_recarga_por_municipio(gdf_filtrado, gdf_municipios)
        guardar_resultados(gdf_final, metricas)

        logger.info("Silver layer cargadores_geometria completed successfully.")


        # ----------------------------------------------------
        # 3. GOLD
        # ----------------------------------------------------

        logger.info("Starting Gold layer...")

        gold = generar_gold(f"{RAW_PATH}", f"{GOLD_PATH}gold.parquet", PESOS_MODELO)
        modelo, baseline = comparar_con_baseline(gold)
        resultados_kmeans, metricas = analizar_kmeans(gold)

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