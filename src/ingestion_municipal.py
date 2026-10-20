"""Ingesta de las fuentes municipales (DE1): DGT, INE Padrón e INE EOH.

Cada función guarda el original SIN modificar en data/raw/ y devuelve su ruta.
Las funciones no reciben argumentos y no ejecutan nada al importarse.
"""
from pathlib import Path

import requests

# Rutas relativas a la raíz del repositorio (este archivo está en src/)
RAIZ = Path(__file__).resolve().parents[1]
RAW_DGT = RAIZ / "data" / "raw" / "dgt"
RAW_INE = RAIZ / "data" / "raw" / "ine"

# Nombres de los originales (silver los lee con estos nombres)
ARCHIVO_DGT = RAW_DGT / "DatosMunicipalesGeneral_2025.xlsx"
ARCHIVO_PADRON = RAW_INE / "padron_29005.json"
ARCHIVO_EOH = RAW_INE / "eoh_2078.csv"

# Fuentes
URL_DGT = (
    "https://www.dgt.es/export/sites/web-DGT/.galleries/downloads/"
    "dgt-en-cifras/informacion_municipal/Datos-municipales-general/"
    "DatosMunicipalesGeneral_2025.xlsx"
)
URL_PADRON = "https://servicios.ine.es/wstempus/jsCache/ES/DATOS_TABLA/29005?tip=AM"
URL_EOH = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/2078.csv"


def descargar_dgt():
    """DGT: Excel de datos municipales 2025. Si ya está en raw, no lo vuelve a descargar."""
    if ARCHIVO_DGT.is_file():
        return ARCHIVO_DGT
    RAW_DGT.mkdir(parents=True, exist_ok=True)
    respuesta = requests.get(URL_DGT, timeout=120)
    respuesta.raise_for_status()
    ARCHIVO_DGT.write_bytes(respuesta.content)
    return ARCHIVO_DGT


def descargar_padron():
    """INE Padrón (tabla 29005) por API, con metadatos (tip=AM). Guarda el JSON original."""
    respuesta = requests.get(URL_PADRON, timeout=300)
    respuesta.raise_for_status()
    # El INE puede responder 200 con un aviso en vez de datos: debe ser una lista de series
    if not isinstance(respuesta.json(), list):
        raise RuntimeError(f"Respuesta inesperada del INE: {respuesta.text[:200]}")
    RAW_INE.mkdir(parents=True, exist_ok=True)
    ARCHIVO_PADRON.write_bytes(respuesta.content)
    return ARCHIVO_PADRON


def descargar_eoh():
    """INE EOH (tabla 2078) en CSV. Se usa el CSV porque trae el código INE de 5 dígitos."""
    respuesta = requests.get(URL_EOH, timeout=120)
    respuesta.raise_for_status()
    RAW_INE.mkdir(parents=True, exist_ok=True)
    ARCHIVO_EOH.write_bytes(respuesta.content)
    return ARCHIVO_EOH