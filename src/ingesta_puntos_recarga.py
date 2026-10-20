from pathlib import Path

import requests


URL_DGT = (
    "https://nap.dgt.es/datex2/v3/miterd/"
    "EnergyInfrastructureTablePublication/electrolineras.xml"
)


def descargar_dgt(carpeta_destino: Path) -> Path:
    """Descarga el XML de puntos de recarga de la DGT."""

    carpeta_destino.mkdir(parents=True, exist_ok=True)

    archivo_destino = carpeta_destino / "electrolineras.xml"

    response = requests.get(
        URL_DGT,
        timeout=(30, 300)
    )
    response.raise_for_status()

    archivo_destino.write_bytes(response.content)

    return archivo_destino