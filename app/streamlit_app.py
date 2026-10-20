import streamlit as st
import pandas as pd
import boto3

from io import BytesIO
import plotly.express as px

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

    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket="sot-01-ev-location",
        Key="gold/municipios.parquet"
    )

    df = pd.read_parquet(
        BytesIO(response["Body"].read())
    )

    return df


df = load_data()

# =====================================================
# FILTROS
# =====================================================

st.sidebar.header("Filtros")

provincias = ["Todas"] + sorted(
    df["provincia"].dropna().unique().tolist()
)

provincia = st.sidebar.selectbox(
    "Provincia",
    provincias
)

score_minimo = st.sidebar.slider(
    "Score mínimo",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

clusters = ["Todos"] + sorted(
    df["cluster"].dropna().unique().tolist()
)

cluster = st.sidebar.selectbox(
    "Cluster",
    clusters
)


# =====================================================
# APLICACIÓN DE FILTROS
# =====================================================

df_filtrado = df.copy()

if provincia != "Todas":
    df_filtrado = df_filtrado[
        df_filtrado["provincia"] == provincia
    ]

if cluster != "Todos":
    df_filtrado = df_filtrado[
        df_filtrado["cluster"] == cluster
    ]

df_filtrado = df_filtrado[
    df_filtrado["score"] >= score_minimo
]


# =====================================================
# KPIs
# =====================================================

st.subheader("Indicadores principales")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Municipios",
    len(df_filtrado)
)

col2.metric(
    "Vehículos eléctricos",
    int(df_filtrado["vehiculos_electricos"].sum())
)

col3.metric(
    "Puntos de recarga",
    int(df_filtrado["puntos_recarga"].sum())
)

col4.metric(
    "Score medio",
    round(df_filtrado["score"].mean(), 2)
)


# =====================================================
# MAPA
# =====================================================

st.subheader("Mapa de oportunidades")

fig_mapa = px.scatter_map(
    df_filtrado,
    lat="latitud",
    lon="longitud",
    color="score",
    size="poblacion",
    hover_name="municipio",
    hover_data=[
        "provincia",
        "score",
        "cluster",
        "puntos_recarga",
        "vehiculos_electricos"
    ],
    color_continuous_scale="RdYlGn",
    zoom=5,
    height=550
)

st.plotly_chart(
    fig_mapa,
    width="stretch"
)


# =====================================================
# RANKING
# =====================================================

st.subheader("Ranking de municipios")

ranking = (
    df_filtrado
    .sort_values(
        by="score",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    ranking[
        [
            "municipio",
            "provincia",
            "poblacion",
            "vehiculos_electricos",
            "puntos_recarga",
            "score",
            "cluster"
        ]
    ],
    width="stretch",
    hide_index=True
)


# =====================================================
# DETALLE DE MUNICIPIO
# =====================================================

st.subheader("Detalle del municipio")

municipios = sorted(
    df_filtrado["municipio"].unique()
)

if municipios:

    municipio_seleccionado = st.selectbox(
        "Selecciona un municipio",
        municipios
    )

    municipio = df_filtrado[
        df_filtrado["municipio"]
        == municipio_seleccionado
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Score",
        round(municipio["score"], 2)
    )

    col2.metric(
        "Población",
        int(municipio["poblacion"])
    )

    col3.metric(
        "Vehículos eléctricos",
        int(municipio["vehiculos_electricos"])
    )

    col4.metric(
        "Puntos de recarga",
        int(municipio["puntos_recarga"])
    )

    st.write(
        f"**Cluster:** {municipio['cluster']}"
    )


# =====================================================
# CLUSTERS
# =====================================================

st.subheader("Segmentación de municipios")

cluster_counts = (
    df_filtrado["cluster"]
    .value_counts()
    .reset_index()
)

cluster_counts.columns = [
    "cluster",
    "municipios"
]

fig_clusters = px.bar(
    cluster_counts,
    x="cluster",
    y="municipios",
    color="cluster",
    title="Número de municipios por cluster"
)

st.plotly_chart(
    fig_clusters,
    width="stretch"
)


# =====================================================
# DATOS
# =====================================================

with st.expander("Ver dataset filtrado"):

    st.dataframe(
        df_filtrado,
        width="stretch",
        hide_index=True
    )