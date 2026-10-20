"""Silver municipal (DE1): tablas limpias a partir de los originales de data/raw/.

Cada función lee lo que necesita de data/raw/, construye su tabla, la guarda en
data/processed/ (Parquet) y la devuelve. No reciben argumentos ni ejecutan nada
al importarse. Requiere haber ejecutado antes la ingesta (ingestion_municipal.py).
"""
import json
from pathlib import Path

import pandas as pd

# Rutas relativas a la raíz del repositorio (este archivo está en src/)
RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"

# Originales generados por la ingesta
ARCHIVO_DGT = RAW / "dgt" / "DatosMunicipalesGeneral_2025.xlsx"
ARCHIVO_PADRON = RAW / "ine" / "padron_29005.json"
ARCHIVO_EOH = RAW / "ine" / "eoh_2078.csv"

# Parámetros del análisis: para cambiar de periodo solo hay que tocar estas líneas
ANIO = 2025
ANIO_BASE = 2020
UMBRAL_POBLACION = 10_000


def _limpiar_nombre(nombres):
    """Quita comillas y espacios o tabuladores de los extremos (caso 21902)."""
    return nombres.str.replace('"', "", regex=False).str.strip(" \t")


def _provincias_dgt():
    """DGT -> diccionario {código de provincia (2 dígitos): nombre de la provincia}."""
    dgt = pd.read_excel(ARCHIVO_DGT, usecols=["Código INE", "Provincia"],
                        dtype={"Código INE": str})
    dgt["cod_provincia"] = dgt["Código INE"].str[:2]
    # Cada código de provincia debe tener un único nombre
    if (dgt.groupby("cod_provincia")["Provincia"].nunique() != 1).any():
        raise ValueError("Hay códigos de provincia con más de un nombre en la DGT")
    return dgt.drop_duplicates("cod_provincia").set_index("cod_provincia")["Provincia"].to_dict()


def _pernoctaciones_eoh():
    """EOH -> pernoctaciones de <ANIO> por cod_ine (España y Extranjero sumados)."""
    eoh = pd.read_csv(ARCHIVO_EOH, sep=";", dtype=str, encoding="utf-8-sig")

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


def construir_municipios():
    """Padrón + DGT + EOH -> municipios: población, provincia, pernoctaciones y mas_10000."""
    with ARCHIVO_PADRON.open("r", encoding="utf-8") as archivo:
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

    # Único cruce por cod_ine: la EOH se añade por left join (no se pierde ningún municipio)
    pernoctaciones = _pernoctaciones_eoh()
    if not pernoctaciones["cod_ine"].isin(municipios["cod_ine"]).all():
        raise ValueError("Hay puntos turísticos de la EOH que no existen en el Padrón")
    municipios = municipios.merge(pernoctaciones, on="cod_ine", how="left", validate="one_to_one")

    # Marca el filtro de tamaño sin eliminar filas; sin población implica False
    municipios["mas_10000"] = municipios[f"pob_{ANIO}"].gt(UMBRAL_POBLACION).fillna(False)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    municipios.to_parquet(PROCESSED / "municipios.parquet", index=False)
    return municipios


def construir_vehiculos():
    """DGT -> vehiculos: parque y distintivos por municipio, con sin_especificar."""
    dgt = pd.read_excel(ARCHIVO_DGT, dtype={"Código INE": str})

    vehiculos = dgt[["Código INE", "Municipio", "Provincia", "Parque Total",
                     "Distintivo 0", "Distintivo ECO"]].rename(columns={
        "Código INE": "cod_ine",
        "Municipio": "nombre",
        "Provincia": "provincia",
        "Parque Total": "parque_total",
        "Distintivo 0": "dist_0",
        "Distintivo ECO": "dist_eco",
    })
    # Las filas "Sin Especificar" se marcan, no se eliminan
    vehiculos["sin_especificar"] = vehiculos["nombre"].str.contains(
        "Sin Especificar", regex=False, na=False)
    vehiculos["nombre"] = _limpiar_nombre(vehiculos["nombre"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    vehiculos.to_parquet(PROCESSED / "vehiculos.parquet", index=False)
    return vehiculos