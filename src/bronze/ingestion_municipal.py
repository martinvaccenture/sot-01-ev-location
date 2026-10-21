"""Ingesta de las fuentes municipales (DE1): DGT, INE Padrón e INE EOH.

Cada función guarda el original SIN modificar en la ruta S3_RAW_PATH del .env.
No reciben argumentos y no devuelven nada.
"""
import os

import fsspec
import requests
from dotenv import load_dotenv

load_dotenv()

# Nombres de los originales (silver los lee con estos nombres)
NOMBRE_DGT = "DatosMunicipalesGeneral_2025.xlsx"
NOMBRE_PADRON = "padron_29005.json"
NOMBRE_EOH = "eoh_2078.csv"

# Fuentes
URL_DGT = (
    "https://www.dgt.es/export/sites/web-DGT/.galleries/downloads/"
    "dgt-en-cifras/informacion_municipal/Datos-municipales-general/"
    "DatosMunicipalesGeneral_2025.xlsx"
)
URL_PADRON = "https://servicios.ine.es/wstempus/jsCache/ES/DATOS_TABLA/29005?tip=AM"
URL_EOH = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/2078.csv"


def _ruta_raw(*partes):
    """Ruta dentro de raw. Falla con un mensaje claro si S3_RAW_PATH no está en el .env."""
    base = os.getenv("S3_RAW_PATH", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Falta S3_RAW_PATH en el .env")
    return "/".join([base, *partes])


def _guardar(contenido, ruta):
    """Guarda bytes en la ruta (S3 o local; en local crea las carpetas)."""
    with fsspec.open(ruta, "wb") as archivo:
        archivo.write(contenido)


def descargar_dgt():
    """DGT: Excel de datos municipales 2025."""
    respuesta = requests.get(URL_DGT, timeout=120)
    respuesta.raise_for_status()
    _guardar(respuesta.content, _ruta_raw("dgt", NOMBRE_DGT))


def descargar_padron():
    """INE Padrón (tabla 29005) por API, con metadatos (tip=AM). Guarda el JSON original."""
    respuesta = requests.get(URL_PADRON, timeout=300)
    respuesta.raise_for_status()
    # El INE puede responder 200 con un aviso en vez de datos: debe ser una lista de series
    if not isinstance(respuesta.json(), list):
        raise RuntimeError(f"Respuesta inesperada del INE: {respuesta.text[:200]}")
    _guardar(respuesta.content, _ruta_raw("ine", NOMBRE_PADRON))


def descargar_eoh():
    """INE EOH (tabla 2078) en CSV. Se usa el CSV porque trae el código INE de 5 dígitos."""
    respuesta = requests.get(URL_EOH, timeout=120)
    respuesta.raise_for_status()
    _guardar(respuesta.content, _ruta_raw("ine", NOMBRE_EOH))
