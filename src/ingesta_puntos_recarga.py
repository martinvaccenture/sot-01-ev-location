import os

import requests
import s3fs
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN
# ============================================================
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
RAW_PATH = os.getenv("S3_RAW_PATH")

# ============================================================
# URL DGT
# ============================================================
URL_DGT = (
    "https://nap.dgt.es/datex2/v3/miterd/"
    "EnergyInfrastructureTablePublication/electrolineras.xml"
)

# ============================================================
# DESCARGA DGT
# ============================================================
def descargar_dgt():
    """
    Descarga el XML de puntos de recarga de la DGT
    y lo guarda directamente en la capa RAW de S3.
    """

    response = requests.get(URL_DGT, timeout=(30, 300))
    response.raise_for_status()

    ruta_destino = (f"{RAW_PATH}"
        "dgt/electrolineras.xml")

    s3 = s3fs.S3FileSystem()

    with s3.open(ruta_destino, "wb") as archivo:
        archivo.write(response.content)