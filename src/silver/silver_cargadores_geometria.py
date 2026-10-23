"""Preprocesado e ingesta geoespacial de puntos de recarga DGT y municipios CNIG."""

import os
import xml.etree.ElementTree as ET
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import s3fs
from dotenv import load_dotenv


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
RAW_PATH = os.getenv("S3_RAW_PATH")
SILVER_PATH = os.getenv("S3_SILVER_PATH")
GOLD_PATH = os.getenv("S3_GOLD_PATH")


# ============================================================
# PREPROCESADO DGT
# ============================================================

def leer_puntos_recarga_xml(ruta_xml):
    """
    Lee el XML de puntos de recarga de la DGT almacenado en S3
    y genera un DataFrame con una fila por conector.
    """

    s3 = s3fs.S3FileSystem()

    with s3.open(ruta_xml, "rb") as archivo:
        root = ET.parse(archivo).getroot()

    filas = []

    for sitio in root.findall("./*{*}energyInfrastructureSite"):

        site_id = sitio.get("id")

        latitud = obtener_texto(sitio, ".//{*}latitude")
        longitud = obtener_texto(sitio, ".//{*}longitude")
        estaciones = sitio.findall(".//{*}energyInfrastructureStation")

        for estacion in estaciones:

            station_id = estacion.get("id")
            puntos = estacion.findall(".//{*}refillPoint")

            for punto in puntos:
                refill_point_id = punto.get("id")
                nombre_punto = obtener_texto(punto,
                    ".//{*}name/{*}values/{*}value")

                conectores = punto.findall(".//{*}connector")

                for indice, conector in enumerate(
                    conectores,
                    start=1):

                    tipo_conector = obtener_texto(conector,
                        ".//{*}connectorType")

                    potencia_w = obtener_texto(conector,
                        ".//{*}maxPowerAtSocket")

                    filas.append({
                        "site_id": site_id,
                        "station_id": station_id,
                        "refill_point_id": refill_point_id,
                        "connector_index": indice,
                        "nombre_punto": nombre_punto,
                        "connector_type": tipo_conector,
                        "latitud": latitud,
                        "longitud": longitud,
                        "potencia_w": potencia_w
                    })

    df = pd.DataFrame(filas)

    df["latitud"] = pd.to_numeric(df["latitud"],
        errors="coerce")

    df["longitud"] = pd.to_numeric(df["longitud"],
        errors="coerce")

    df["potencia_w"] = pd.to_numeric(df["potencia_w"],
        errors="coerce")

    df["potencia_kw"] = (df["potencia_w"] / 1000)

    return df


def obtener_texto(elemento, ruta):
    """
    Obtiene el texto de una etiqueta XML.
    """

    if elemento is None:
        return None

    nodo = elemento.find(ruta)

    if nodo is None or nodo.text is None:
        return None

    return nodo.text.strip()


# ============================================================
# GEOMETRÍAS DGT
# ============================================================

def convertir_coordenadas_en_geometrias(df):
    """
    Convierte longitud y latitud en geometrías Point.
    """

    gdf_recarga = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["longitud"],
            df["latitud"]
        ),
        crs="EPSG:4326"
    )

    return gdf_recarga


# ============================================================
# PREPROCESADO MUNICIPIOS CNIG
# ============================================================

def leer_shapefile_s3(ruta_zip, nombre_shp):
    """
    Lee un Shapefile contenido en un ZIP almacenado en S3.
    """

    s3 = s3fs.S3FileSystem()

    with tempfile.TemporaryDirectory() as carpeta_temporal:

        ruta_zip_temporal = (
            Path(carpeta_temporal)
            / "archivo.zip")

        s3.get(ruta_zip,
            str(ruta_zip_temporal))

        with zipfile.ZipFile(
            ruta_zip_temporal,
            "r") as archivo_zip:

            archivo_zip.extractall(
                carpeta_temporal)

        ruta_shp = (
            Path(carpeta_temporal)
            / nombre_shp)

        gdf = gpd.read_file(
            ruta_shp)

    return gdf


def preparar_municipios():
    """
    Lee desde S3 los municipios de Península/Baleares y Canarias,
    unifica el CRS y genera un único GeoDataFrame.
    """

    ruta_peninbal = (f"{RAW_PATH}"
        "cnig/recintos_municipales_peninbal.zip")

    ruta_canarias = (f"{RAW_PATH}"
        "cnig/recintos_municipales_canarias.zip")

    gdf_peninbal = leer_shapefile_s3(ruta_peninbal,
        "recintos_municipales_inspire_peninbal_etrs89.shp")

    gdf_canarias = leer_shapefile_s3(ruta_canarias,
        "recintos_municipales_inspire_canarias_wgs84.shp")

    gdf_peninbal = gdf_peninbal.to_crs("EPSG:4326")

    gdf_canarias = gdf_canarias.to_crs("EPSG:4326")

    columnas_municipio = [
        "NATCODE",
        "NAMEUNIT",
        "geometry"
    ]

    gdf_peninbal = gdf_peninbal[columnas_municipio].copy()

    gdf_canarias = gdf_canarias[columnas_municipio].copy()

    gdf_municipios = gpd.GeoDataFrame(
        pd.concat(
            [
                gdf_peninbal,
                gdf_canarias
            ],
            ignore_index=True
        ),
        geometry="geometry",
        crs="EPSG:4326"
    )

    return gdf_municipios


# ============================================================
# SPATIAL JOIN
# ============================================================

def realizar_spatial_join(
    gdf_recarga,
    gdf_municipios
):
    """
    Asigna cada conector al municipio en el que se encuentra.
    """

    if gdf_recarga.crs != gdf_municipios.crs:

        gdf_recarga = gdf_recarga.to_crs(
            gdf_municipios.crs
        )

    gdf_join = gpd.sjoin(
        gdf_recarga,
        gdf_municipios[
            [
                "NATCODE",
                "NAMEUNIT",
                "geometry"
            ]
        ],
        how="left",
        predicate="within"
    )

    return gdf_join


# ============================================================
# CRUCES FALLIDOS
# ============================================================

def medir_cruces_fallidos(gdf_join):
    """
    Calcula los cruces correctos y fallidos del spatial join.
    """

    total = len(gdf_join)

    cruces_correctos = (gdf_join["NATCODE"]
        .notna()
        .sum()
    )

    cruces_fallidos = (gdf_join["NATCODE"]
        .isna()
        .sum()
    )

    if total > 0:

        porcentaje_correctos = (
            cruces_correctos / total * 100)

        porcentaje_fallidos = (
            cruces_fallidos / total * 100)

    else:

        porcentaje_correctos = 0
        porcentaje_fallidos = 0

    metricas = {
        "total": int(total),
        "cruces_correctos": int(cruces_correctos),
        "cruces_fallidos": int(cruces_fallidos),
        "porcentaje_correctos": float(
            porcentaje_correctos
        ),
        "porcentaje_fallidos": float(
            porcentaje_fallidos
        )
    }

    gdf_cruces_fallidos = gdf_join[
        gdf_join["NATCODE"].isna()
    ].copy()

    return metricas, gdf_cruces_fallidos


# ============================================================
# FILTRO >= 50 KW
# ============================================================

def filtrar_recarga_minima(
    gdf_join,
    potencia_minima_kw=50
):
    """
    Filtra conectores cuya potencia sea igual o superior
    al umbral indicado.
    """

    gdf_filtrado = gdf_join[
        gdf_join["potencia_kw"]
        >= potencia_minima_kw
    ].copy()

    return gdf_filtrado


# ============================================================
# AGREGACIÓN POR MUNICIPIO
# ============================================================

def agregar_recarga_por_municipio(
    gdf_filtrado,
    gdf_municipios
):
    """
    Agrega sites, estaciones, puntos de recarga y conectores
    >= 50 kW por municipio.
    """

    df_agregado = (
        gdf_filtrado
        .dropna(subset=["NATCODE"])
        .groupby(
            [
                "NATCODE",
                "NAMEUNIT"
            ]
        )
        .agg(
            sites_50kw=("site_id","nunique"),
            stations_50kw=("station_id","nunique"),
            refill_points_50kw=("refill_point_id","nunique"),
            connectors_50kw=("connector_index","size")
        )
        .reset_index()
    )

    gdf_final = gdf_municipios.merge(
        df_agregado,
        on=[
            "NATCODE",
            "NAMEUNIT"
        ],
        how="left"
    )

    columnas_conteo = [
        "sites_50kw",
        "stations_50kw",
        "refill_points_50kw",
        "connectors_50kw"
    ]

    gdf_final[columnas_conteo] = (
        gdf_final[columnas_conteo]
        .fillna(0)
        .astype(int)
    )

    gdf_final = gpd.GeoDataFrame(
        gdf_final,
        geometry="geometry",
        crs=gdf_municipios.crs
    )

    return gdf_final


# ============================================================
# GUARDAR RESULTADOS EN S3
# ============================================================

def guardar_resultados(
    gdf_final,
    gdf_cruces_fallidos
):
    """
    Guarda los resultados directamente en la capa GOLD de S3.
    """

    ruta_resultado = (f"{SILVER_PATH}"
        "puntos_recarga.parquet")

    ruta_fallidos = (f"{SILVER_PATH}"
        "cruces_fallidos.parquet")

    gdf_final.to_parquet(
        ruta_resultado,
        index=False)

    gdf_cruces_fallidos.to_parquet(
        ruta_fallidos,
        index=False)