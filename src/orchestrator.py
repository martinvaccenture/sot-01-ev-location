import logging
import sys
from datetime import datetime

from ingesta_puntos_recarga import descargar_dgt as descargar_dgt_puntos
from ingestion_municipal import *
from silver_municipal import *
from gold import run_gold


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
        


        # ----------------------------------------------------
        # 3. GOLD
        # ----------------------------------------------------

        logger.info("Starting Gold layer...")

        run_gold()

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