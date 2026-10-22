import streamlit as st
import pandas as pd
import boto3

from io import BytesIO
import plotly.express as px
import os
from dotenv import load_dotenv

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

# =====================================================
# CARGA DE DATOS DESDE S3
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_parquet(
        f"{GOLD_PATH}gold.parquet"
    )

    return df

df = load_data()

# ============================================================

# FILTROS

# ============================================================
 
st.sidebar.header("Filtros")
 
provincias = ["Todas"] + sorted(

    df["provincia"]

    .dropna()

    .unique()

    .tolist()

)
 
provincia = st.sidebar.selectbox(

    "Provincia",

    provincias

)
 
 
score_max = float(df["opportunity_score"].max())
 
score_minimo = st.sidebar.slider(

    "Opportunity Score mínimo",

    min_value=0.0,

    max_value=score_max,

    value=0.0,

    step=0.01

)
 
 
ranking_max = int(df["ranking_general"].max())
 
top_n = st.sidebar.slider(

    "Número de municipios en el ranking",

    min_value=5,

    max_value=min(50, ranking_max),

    value=min(10, ranking_max),

    step=5

)
 
 
# ============================================================

# APLICACIÓN DE FILTROS

# ============================================================
 
df_filtrado = df.copy()
 
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

# RANKING

# ============================================================
 
st.divider()
 
st.subheader("🏆 Ranking de oportunidades")
 
 
ranking = (

    df_filtrado

    .sort_values(

        by="ranking_general",

        ascending=True

    )

    .head(top_n)

)
 
 
ranking_mostrar = ranking[

    [

        "ranking_general",

        "municipio",

        "provincia",

        "poblacion",

        "vehiculos_cero_eco",

        "puntos_rapidos",

        "opportunity_score"

    ]

].copy()
 
 
ranking_mostrar.columns = [

    "Ranking",

    "Municipio",

    "Provincia",

    "Población",

    "Vehículos Cero/ECO",

    "Puntos rápidos",

    "Opportunity Score"

]
 
 
st.dataframe(

    ranking_mostrar,

    hide_index=True,

    width="stretch"

)
 
 
# ============================================================

# GRÁFICO DEL TOP

# ============================================================
 
fig_ranking = px.bar(

    ranking.sort_values(

        "opportunity_score",

        ascending=True

    ),

    x="opportunity_score",

    y="municipio",

    orientation="h",

    color="opportunity_score",

    color_continuous_scale="RdYlGn",

    labels={

        "opportunity_score": "Opportunity Score",

        "municipio": "Municipio"

    },

    title=f"Top {len(ranking)} municipios por Opportunity Score"

)
 
 
fig_ranking.update_layout(

    coloraxis_showscale=False,

    yaxis_title=None,

    xaxis_title="Opportunity Score"

)
 
 
st.plotly_chart(

    fig_ranking,

    width="stretch"

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
 
 
municipio = df_filtrado[

    df_filtrado["municipio"] == municipio_seleccionado

].sort_values(

    "ranking_general"

).iloc[0]
 
 
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

    "Puntos rápidos por 10.000 hab.",

    f"{municipio['puntos_rapidos_por_10000']:.2f}"

)
 
 
# ============================================================

# VARIABLES DEL SCORE

# ============================================================
 
st.subheader("Factores que explican la oportunidad")
 
 
factores = pd.DataFrame(

    {

        "Factor": [

            "Vehículos por habitante",

            "Cero/ECO por 1.000 habitantes",

            "Puntos rápidos por 10.000 habitantes",

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

# COMPARACIÓN DE SCORES

# ============================================================
 
st.divider()
 
st.subheader("📊 Distribución del Opportunity Score")
 
 
fig_score = px.histogram(

    df_filtrado,

    x="opportunity_score",

    nbins=25,

    labels={

        "opportunity_score": "Opportunity Score"

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

# DATASET COMPLETO

# ============================================================
 
with st.expander("Ver dataset Gold filtrado"):
 
    st.dataframe(

        df_filtrado.sort_values(

            "ranking_general"

        ),

        hide_index=True,

        width="stretch"

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