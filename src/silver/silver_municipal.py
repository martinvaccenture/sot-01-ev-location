"""Silver municipal (DE1): tablas limpias a partir de los originales de raw.

Cada función lee de S3_RAW_PATH y guarda en S3_SILVER_PATH (Parquet), según el .env.
No reciben argumentos y no devuelven nada. Requiere haber ejecutado antes la ingesta.
Las dos tablas contienen solo los municipios con más de UMBRAL_POBLACION habitantes.
"""
import json
import os

import fsspec
import pandas as pd
from dotenv import load_dotenv

# Variables del .env: S3_RAW_PATH, S3_SILVER_PATH y AWS_REGION
load_dotenv()

# Originales generados por la ingesta
NOMBRE_DGT = "DatosMunicipalesGeneral_2025.xlsx"
NOMBRE_PADRON = "padron_29005.json"
NOMBRE_EOH = "eoh_2078.csv"

# Parámetros del análisis: para cambiar de periodo solo hay que tocar estas líneas
ANIO = 2025
ANIO_BASE = 2020
UMBRAL_POBLACION = 10_000


def _ruta(variable, *partes):
    """Ruta dentro de raw o silver. Falla con un mensaje claro si la variable no está en el .env."""
    base = os.getenv(variable, "").strip().rstrip("/")
    if not base:
        raise RuntimeError(f"Falta {variable} en el .env")
    return "/".join([base, *partes])


def _guardar_parquet(tabla, nombre):
    """Guarda la tabla en silver como Parquet, sin índice (S3 o local; en local crea las carpetas)."""
    with fsspec.open(_ruta("S3_SILVER_PATH", nombre), "wb") as archivo:
        tabla.to_parquet(archivo, index=False)


def _limpiar_nombre(nombres):
    """Quita comillas y espacios o tabuladores de los extremos (caso 21902)."""
    return nombres.str.replace('"', "", regex=False).str.strip(" \t")


def _provincias_dgt():
    """DGT -> diccionario {código de provincia (2 dígitos): nombre de la provincia}."""
    dgt = pd.read_excel(_ruta("S3_RAW_PATH", "dgt", NOMBRE_DGT),
                        usecols=["Código INE", "Provincia"], dtype={"Código INE": str})
    dgt["cod_provincia"] = dgt["Código INE"].str[:2]
    # Cada código de provincia debe tener un único nombre
    if (dgt.groupby("cod_provincia")["Provincia"].nunique() != 1).any():
        raise ValueError("Hay códigos de provincia con más de un nombre en la DGT")
    return dgt.drop_duplicates("cod_provincia").set_index("cod_provincia")["Provincia"].to_dict()


def _pernoctaciones_eoh():
    """EOH -> pernoctaciones de <ANIO> por cod_ine (España y Extranjero sumados)."""
    eoh = pd.read_csv(_ruta("S3_RAW_PATH", "ine", NOMBRE_EOH),
                      sep=";", dtype=str, encoding="utf-8-sig")

    # Código INE: los 5 primeros dígitos de "Puntos turísticos"
    eoh["cod_ine"] = eoh["Puntos turísticos"].str.extract(r"^([0-9]{5})", expand=False)

    # "Total" a número: ".", ".." y vacío -> NaN (nunca 0); el punto es separador de miles
    total = eoh["Total"]
    con_dato = total.notna() & ~total.isin([".", "..", ""])
    if not total[con_dato].str.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{3})*|[0-9]+").all():
        raise ValueError("Total tiene un formato numérico inesperado")
    eoh["Total"] = pd.to_numeric(total.where(con_dato).str.replace(".", "", regex=False))

    # Solo pernoctaciones del año indicado
    eoh = eoh[eoh["Viajeros y pernoctaciones"].eq("Pernoctaciones")
              & eoh["Periodo"].str.startswith(f"{ANIO}M", na=False)]

    # Suma de meses y de ambas residencias; sin ningún dato queda NaN (no 0)
    return (eoh.groupby("cod_ine", as_index=False)["Total"].sum(min_count=1)
               .rename(columns={"Total": f"pernoctaciones_{ANIO}"}))


def _municipios_completos():
    """Padrón + provincia (DGT) + EOH -> todos los municipios, SIN filtrar por población."""
    with fsspec.open(_ruta("S3_RAW_PATH", "ine", NOMBRE_PADRON), "rb") as archivo:
        series = json.load(archivo)

    filas = []
    for serie in series:
        metadatos = {meta["T3_Variable"]: meta for meta in serie["MetaData"]}
        # Solo la serie Sexo = Total; se ignoran Hombres y Mujeres
        if metadatos["Sexo"]["Nombre"] != "Total":
            continue
        municipio = metadatos["Municipios"]
        poblacion = {dato["Anyo"]: dato["Valor"] for dato in serie["Data"]}
        # Un año ausente queda como NaN (nunca 0)
        filas.append({
            "cod_ine": str(municipio["Codigo"]),
            "nombre": municipio["Nombre"],
            f"pob_{ANIO}": poblacion.get(ANIO, float("nan")),
            f"pob_{ANIO_BASE}": poblacion.get(ANIO_BASE, float("nan")),
        })

    municipios = pd.DataFrame(filas)
    municipios["nombre"] = _limpiar_nombre(municipios["nombre"])

    # Provincia: los 2 primeros dígitos del código INE, con el nombre sacado de la DGT
    municipios.insert(2, "provincia", municipios["cod_ine"].str[:2].map(_provincias_dgt()))

    # Cruce por cod_ine: la EOH se añade por left join (no se pierde ningún municipio).
    # La comprobación va ANTES del filtro: la EOH incluye puntos de menos de 10.000 habitantes.
    pernoctaciones = _pernoctaciones_eoh()
    if not pernoctaciones["cod_ine"].isin(municipios["cod_ine"]).all():
        raise ValueError("Hay puntos turísticos de la EOH que no existen en el Padrón")
    return municipios.merge(pernoctaciones, on="cod_ine", how="left", validate="one_to_one")


def _solo_mayores_del_umbral(municipios):
    """Filtro del caso: solo municipios con más de UMBRAL_POBLACION habitantes en <ANIO>.
    Sin población de ese año no pasan el filtro (un NaN no es mayor que el umbral)."""
    return municipios[municipios[f"pob_{ANIO}"].gt(UMBRAL_POBLACION)].reset_index(drop=True)


def construir_municipios():
    """Padrón + DGT + EOH -> silver/municipios.parquet (más de 10.000 habitantes)."""
    _guardar_parquet(_solo_mayores_del_umbral(_municipios_completos()), "municipios.parquet")


def construir_vehiculos():
    """DGT -> silver/vehiculos.parquet (parque y distintivos de los municipios del filtro)."""
    dgt = pd.read_excel(_ruta("S3_RAW_PATH", "dgt", NOMBRE_DGT), dtype={"Código INE": str})

    vehiculos = dgt[["Código INE", "Municipio", "Provincia", "Parque Total",
                     "Distintivo 0", "Distintivo ECO"]].rename(columns={
        "Código INE": "cod_ine",
        "Municipio": "nombre",
        "Provincia": "provincia",
        "Parque Total": "parque_total",
        "Distintivo 0": "dist_0",
        "Distintivo ECO": "dist_eco",
    })
    vehiculos["nombre"] = _limpiar_nombre(vehiculos["nombre"])

    # Los municipios del filtro se deciden con la población del Padrón
    codigos = set(_solo_mayores_del_umbral(_municipios_completos())["cod_ine"])
    faltan = codigos - set(vehiculos["cod_ine"])
    if faltan:
        raise ValueError(f"Municipios del filtro que no están en la DGT: {sorted(faltan)}")
    vehiculos = vehiculos[vehiculos["cod_ine"].isin(codigos)].reset_index(drop=True)

    _guardar_parquet(vehiculos, "vehiculos.parquet")
