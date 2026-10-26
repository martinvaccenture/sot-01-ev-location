import sys
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv


# ============================================================
# RUTA RAÍZ DEL PROYECTO
# ============================================================

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from config_modelo import ESCENARIOS


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

GOLD_PATH = os.getenv("S3_GOLD_PATH")

st.set_page_config(
    page_title="EV Location Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Location Intelligence")

st.markdown(
    """
    Priorización de municipios españoles para la instalación
    de hubs de recarga rápida de vehículos eléctricos.
    """
)


# ============================================================
# CARGA DE DATOS DESDE S3
# ============================================================

@st.cache_data
def load_parquet(nombre_archivo):

    return pd.read_parquet(
        f"{GOLD_PATH}{nombre_archivo}"
    )


df_gold = load_parquet(
    "gold.parquet"
)


# ============================================================
# FILTROS
# ============================================================

st.sidebar.header("Filtros")


provincias = ["Todas"] + sorted(
    df_gold["provincia"]
    .dropna()
    .unique()
    .tolist()
)


provincia = st.sidebar.selectbox(
    "Provincia",
    provincias
)


score_max = float(
    df_gold["opportunity_score"].max()
)


score_minimo = st.sidebar.slider(
    "Opportunity Score mínimo",
    min_value=0.0,
    max_value=score_max,
    value=0.0,
    step=0.01
)


ranking_max = int(
    df_gold["ranking_general"].max()
)


top_n = st.sidebar.slider(
    "Número de municipios en el ranking",
    min_value=5,
    max_value=min(50, ranking_max),
    value=min(20, ranking_max),
    step=5
)


valoracion_turismo = st.sidebar.checkbox(
    "Valoración turismo",
    value=False
)


# ============================================================
# FILTRADO DEL GOLD
# ============================================================

df_filtrado = df_gold.copy()


if provincia != "Todas":

    df_filtrado = df_filtrado[
        df_filtrado["provincia"] == provincia
    ]


df_filtrado = df_filtrado[
    df_filtrado["opportunity_score"] >= score_minimo
]


# ============================================================
# COMPROBACIÓN
# ============================================================

if df_filtrado.empty:

    st.warning(
        "No existen municipios que cumplan los filtros seleccionados."
    )

    st.stop()


# ============================================================
# KPIs
# ============================================================

st.subheader("Indicadores principales")

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Municipios",
    f"{len(df_filtrado):,}"
)


col2.metric(
    "Población",
    f"{int(df_filtrado['poblacion'].sum()):,}"
)


col3.metric(
    "Vehículos Cero/ECO",
    f"{int(df_filtrado['vehiculos_cero_eco'].sum()):,}"
)


col4.metric(
    "Puntos de recarga rápida",
    f"{int(df_filtrado['puntos_rapidos'].sum()):,}"
)


# ============================================================
# RANKING DE ESCENARIOS
# ============================================================

st.divider()

st.subheader("🏆 Ranking de oportunidades")


escenario = st.selectbox(
    "Selecciona el modelo",
    [
        "Modelo base",
        "Electrificación",
        "Infraestructura",
        "Crecimiento"
    ]
)


# ============================================================
# CORRESPONDENCIA ESCENARIOS
# ============================================================

nombres_escenarios = {
    "Modelo base": "modelo_base",
    "Electrificación": "electrificacion",
    "Infraestructura": "infraestructura",
    "Crecimiento": "crecimiento"
}


nombre_escenario = nombres_escenarios[
    escenario
]


config_escenario = ESCENARIOS[
    nombre_escenario
]


# ============================================================
# TURISMO / SIN TURISMO
# ============================================================

if valoracion_turismo:

    nombre_archivo_escenario = (
        f"{nombre_escenario}_turismo.parquet"
    )

else:

    nombre_archivo_escenario = (
        f"{nombre_escenario}.parquet"
    )


# ============================================================
# CARGA DEL ESCENARIO
# ============================================================

df_escenario = load_parquet(
    nombre_archivo_escenario
)


# ============================================================
# FILTROS DEL ESCENARIO
# ============================================================

if provincia != "Todas":

    df_escenario = df_escenario[
        df_escenario["provincia"] == provincia
    ]


df_escenario = df_escenario[
    df_escenario["score"] >= score_minimo
]


# ============================================================
# MENSAJE TURISMO
# ============================================================

if valoracion_turismo:

    st.info(
        "🏨 Valoración de turismo activada. "
        f"El escenario {escenario} incorpora la valoración "
        "de las pernoctaciones turísticas cuando existe información."
    )


# ============================================================
# TOP N ESCENARIO
# ============================================================

ranking = (
    df_escenario
    .sort_values(
        "ranking_general",
        ascending=True
    )
    .head(top_n)
    .copy()
)


# ============================================================
# COLUMNAS DEL RANKING
# ============================================================

columnas_ranking = [
    "ranking_general",
    "municipio",
    "provincia",
    "vehiculos_por_habitante",
    "cero_eco_por_1000",
    "puntos_rapidos_por_10000",
    "crecimiento_poblacion"
]


if valoracion_turismo:

    columnas_ranking.append(
        "pernoctaciones_2025"
    )


columnas_ranking.extend(
    [
        "riesgo_flota",
        "score"
    ]
)


ranking_mostrar = ranking[
    columnas_ranking
].copy()


# ============================================================
# NOMBRES DE LAS COLUMNAS
# ============================================================

ranking_mostrar = ranking_mostrar.rename(
    columns={
        "ranking_general": "Ranking",
        "municipio": "Municipio",
        "provincia": "Provincia",
        "vehiculos_por_habitante": "Vehículos / habitante",
        "cero_eco_por_1000": "CERO/ECO / 1.000 hab.",
        "puntos_rapidos_por_10000": "Recarga rápida / 10.000 veh.",
        "crecimiento_poblacion": "Crecimiento población (%)",
        "pernoctaciones_2025": "Pernoctaciones 2025",
        "riesgo_flota": "Riesgo de flota",
        "score": "Opportunity Score"
    }
)


# ============================================================
# FORMATO DE LAS COLUMNAS
# ============================================================

ranking_mostrar = ranking_mostrar.round(
    {
        "Vehículos / habitante": 3,
        "CERO/ECO / 1.000 hab.": 2,
        "Recarga rápida / 10.000 veh.": 2,
        "Crecimiento población (%)": 2,
        "Opportunity Score": 2
    }
)


# ============================================================
# TABLA DEL ESCENARIO
# ============================================================

st.dataframe(
    ranking_mostrar,
    hide_index=True,
    width="stretch"
)


# ============================================================
# PESOS DEL ESCENARIO
# ============================================================

st.subheader("⚖️ Pesos utilizados")


pesos = config_escenario["pesos"]

peso_turismo = config_escenario[
    "peso_turismo"
]


if valoracion_turismo:

    pesos_efectivos = [
        peso * (1 - peso_turismo)
        for peso in pesos
    ]

    df_pesos = pd.DataFrame(
        {
            "Factor": [
                "Vehículos por habitante",
                "CERO/ECO por 1.000 habitantes",
                "Escasez de recarga rápida",
                "Crecimiento de población",
                "Turismo"
            ],
            "Peso": [
                pesos_efectivos[0],
                pesos_efectivos[1],
                pesos_efectivos[2],
                pesos_efectivos[3],
                peso_turismo
            ],
            "Referencia": [
                "Vehículos totales en relación con la población",
                "Vehículos CERO/ECO por 1.000 habitantes",
                "Escasez relativa de puntos de recarga rápida",
                "Evolución de la población del municipio",
                "Pernoctaciones turísticas de 2025"
            ]
        }
    )


else:

    df_pesos = pd.DataFrame(
        {
            "Factor": [
                "Vehículos por habitante",
                "CERO/ECO por 1.000 habitantes",
                "Escasez de recarga rápida",
                "Crecimiento de población"
            ],
            "Peso": pesos,
            "Referencia": [
                "Vehículos totales en relación con la población",
                "Vehículos CERO/ECO por 1.000 habitantes",
                "Escasez relativa de puntos de recarga rápida",
                "Evolución de la población del municipio"
            ]
        }
    )


df_pesos["Peso"] = (
    df_pesos["Peso"]
    .mul(100)
    .round(1)
    .astype(str)
    + " %"
)


st.dataframe(
    df_pesos,
    hide_index=True,
    width="stretch"
)


if valoracion_turismo:

    st.caption(
        "El peso de turismo se aplica únicamente a municipios "
        "con datos de pernoctaciones. Para los municipios sin "
        "información turística se conserva el score calculado "
        "con los cuatro indicadores principales."
    )


# ============================================================
# DETALLE MUNICIPIO
# ============================================================

st.divider()

st.subheader("🔎 Detalle del municipio")


municipios = (
    df_filtrado
    .sort_values("ranking_general")["municipio"]
    .dropna()
    .unique()
    .tolist()
)


municipio_seleccionado = st.selectbox(
    "Selecciona un municipio",
    municipios
)


municipio = (
    df_filtrado[
        df_filtrado["municipio"]
        == municipio_seleccionado
    ]
    .sort_values("ranking_general")
    .iloc[0]
)


# ============================================================
# KPIs MUNICIPIO
# ============================================================

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Ranking general",
    f"#{int(municipio['ranking_general'])}"
)


col2.metric(
    "Opportunity Score",
    f"{municipio['opportunity_score']:.3f}"
)


col3.metric(
    "Población",
    f"{int(municipio['poblacion']):,}"
)


col4.metric(
    "Puntos rápidos",
    f"{int(municipio['puntos_rapidos']):,}"
)


# ============================================================
# SEGUNDA FILA MUNICIPIO
# ============================================================

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Vehículos",
    f"{int(municipio['vehiculos']):,}"
)


col2.metric(
    "Vehículos Cero/ECO",
    f"{int(municipio['vehiculos_cero_eco']):,}"
)


col3.metric(
    "Cero/ECO por 1.000 hab.",
    f"{municipio['cero_eco_por_1000']:.2f}"
)


col4.metric(
    "Puntos rápidos por 10.000 vehículos",
    f"{municipio['puntos_rapidos_por_10000']:.2f}"
)


# ============================================================
# FACTORES DEL SCORE
# ============================================================

st.subheader(
    "Factores que explican la oportunidad"
)


factores = pd.DataFrame(
    {
        "Factor": [
            "Vehículos por habitante",
            "Cero/ECO por 1.000 habitantes",
            "Escasez de recarga rápida",
            "Crecimiento de población"
        ],
        "Valor normalizado": [
            municipio["vehiculos_por_habitante_norm"],
            municipio["cero_eco_por_1000_norm"],
            municipio["puntos_rapidos_por_10000_norm"],
            municipio["crecimiento_poblacion_norm"]
        ]
    }
)


fig_factores = px.bar(
    factores,
    x="Valor normalizado",
    y="Factor",
    orientation="h",
    color="Valor normalizado",
    color_continuous_scale="RdYlGn",
    range_x=[0, 1]
)


fig_factores.update_layout(
    coloraxis_showscale=False,
    xaxis_title="Valor normalizado",
    yaxis_title=None
)


st.plotly_chart(
    fig_factores,
    width="stretch"
)


# ============================================================
# INFORMACIÓN ADICIONAL
# ============================================================

st.subheader("Información adicional")


info = pd.DataFrame(
    {
        "Indicador": [
            "Código INE",
            "Provincia",
            "Población actual",
            "Población hace 5 años",
            "Crecimiento de población",
            "Pernoctaciones 2025",
            "Vehículos por habitante"
        ],
        "Valor": [
            municipio["codigo_ine"],
            municipio["provincia"],
            municipio["poblacion"],
            municipio["poblacion_5y"],
            municipio["crecimiento_poblacion"],
            municipio["pernoctaciones_2025"],
            municipio["vehiculos_por_habitante"]
        ]
    }
)


st.dataframe(
    info,
    hide_index=True,
    width="stretch"
)


# ============================================================
# DISTRIBUCIÓN DEL OPPORTUNITY SCORE
# ============================================================

st.divider()

st.subheader(
    "📊 Distribución del Opportunity Score"
)


fig_score = px.histogram(
    df_filtrado,
    x="opportunity_score",
    nbins=25,
    labels={
        "opportunity_score":
            "Opportunity Score"
    }
)


fig_score.update_layout(
    yaxis_title="Número de municipios"
)


st.plotly_chart(
    fig_score,
    width="stretch"
)


# ============================================================
# GOLD COMPLETO
# ============================================================

with st.expander(
    "Ver dataset Gold filtrado"
):

    st.dataframe(
        df_filtrado.sort_values(
            "ranking_general"
        ),
        hide_index=True,
        width="stretch"
    )


# ============================================================
# RESULTADO FINAL
# ============================================================

st.divider()

st.subheader("✅ Resultado final")


tabla_final = st.selectbox(
    "Selecciona el resultado final",
    [
        "Top 20",
        "Top 20 de confianza"
    ]
)


# ============================================================
# ELECCIÓN DEL PARQUET FINAL
# ============================================================

if tabla_final == "Top 20":

    if valoracion_turismo:

        archivo_final = (
            "top_20_turismo.parquet"
        )

    else:

        archivo_final = (
            "top_20.parquet"
        )


else:

    if valoracion_turismo:

        archivo_final = (
            "top_20_confianza_turismo.parquet"
        )

    else:

        archivo_final = (
            "top_20_confianza.parquet"
        )


# ============================================================
# CARGA TABLA FINAL
# ============================================================

df_final = load_parquet(
    archivo_final
)


if provincia != "Todas":

    df_final = df_final[
        df_final["provincia"] == provincia
    ]


# ============================================================
# COLUMNAS TABLA FINAL
# ============================================================

columnas_finales = [
    "ranking_20",
    "municipio",
    "provincia",
    "poblacion",
    "riesgo_flota",
    "apariciones",
    "aparicion_escenarios",
    "score_medio",
    "score_medio_turismo"
]


columnas_finales = [
    columna
    for columna in columnas_finales
    if columna in df_final.columns
]


df_final_mostrar = df_final[
    columnas_finales
].copy()


df_final_mostrar = df_final_mostrar.rename(
    columns={
        "ranking_20": "Ranking",
        "municipio": "Municipio",
        "provincia": "Provincia",
        "poblacion": "Población",
        "riesgo_flota": "Riesgo de flota",
        "apariciones": "Apariciones",
        "aparicion_escenarios":
            "Proporción de apariciones",
        "score_medio":
            "Score medio",
        "score_medio_turismo":
            "Score medio con turismo"
    }
)


st.dataframe(
    df_final_mostrar,
    hide_index=True,
    width="stretch"
)


# ============================================================
# EXPLICACIÓN RESULTADO
# ============================================================

if tabla_final == "Top 20":

    st.caption(
        "El Top 20 puede contener municipios "
        "identificados con riesgo de flota. "
        "Consulta la columna 'Riesgo de flota' "
        "antes de tomar la decisión final."
    )


else:

    st.success(
        "🛡️ Ranking de confianza: se muestran únicamente "
        "los municipios que han superado el filtro de riesgo "
        "de flota definido por el modelo."
    )


if valoracion_turismo:

    st.info(
        "🏨 Valoración de turismo activada. "
        "El resultado final incorpora los escenarios "
        "con valoración turística."
    )


# ============================================================
# BASELINE
# ============================================================

st.divider()

st.subheader("📍 Baseline por población")

st.markdown(
    """
    Como referencia, se muestran los 20 municipios seleccionados
    únicamente por población, manteniendo un máximo de
    3 municipios por provincia.
    """
)


baseline = (
    df_gold
    .sort_values(
        [
            "poblacion",
            "codigo_ine"
        ],
        ascending=[
            False,
            True
        ]
    )
    .copy()
)


baseline["numero_provincia"] = (
    baseline
    .groupby("provincia")
    .cumcount()
    + 1
)


baseline = (
    baseline[
        baseline["numero_provincia"] <= 3
    ]
    .head(20)
    .copy()
)


baseline.insert(
    0,
    "ranking_baseline",
    range(
        1,
        len(baseline) + 1
    )
)


baseline_mostrar = baseline[
    [
        "ranking_baseline",
        "municipio",
        "provincia",
        "poblacion",
        "vehiculos_cero_eco",
        "puntos_rapidos",
        "opportunity_score"
    ]
].copy()


baseline_mostrar.columns = [
    "Ranking",
    "Municipio",
    "Provincia",
    "Población",
    "Vehículos Cero/ECO",
    "Puntos rápidos",
    "Opportunity Score"
]


st.dataframe(
    baseline_mostrar,
    hide_index=True,
    width="stretch"
)


st.caption(
    "El baseline ordena los municipios exclusivamente por población. "
    "El Opportunity Score se muestra como referencia para facilitar "
    "la comparación con el resultado generado por el modelo."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EV Location Intelligence | "
    "Priorización de municipios para infraestructura "
    "de recarga rápida."
)