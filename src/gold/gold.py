
"""Tabla Gold y análisis del proyecto EV Location Intelligence."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Configuración de gráficos.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

INDICADORES = [
    "vehiculos_por_habitante",
    "cero_eco_por_1000",
    "puntos_rapidos_por_10000",
    "crecimiento_poblacion",
]

ETIQUETAS = [
    "Vehículos / habitante",
    "CERO/ECO / 1.000 habitantes",
    "Recarga rápida / 10.000 vehículos",
    "Crecimiento de población (%)",
]

def seccion(titulo):
    print(f"\n{'=' * 75}\n{titulo}\n{'=' * 75}")


def tabla(datos, columnas=None):
    if columnas is not None:
        datos = datos[columnas]
    print(datos.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ============================================================
# 1. CREACIÓN DE GOLD
# ============================================================

def cargar_datos(carpeta):

    m = pd.read_parquet(carpeta + "municipios.parquet").rename(columns={
        "cod_ine": "codigo_ine",
        "nombre": "municipio",
        "pob_2025": "poblacion",
        "pob_2020": "poblacion_5y",
    })

    v = pd.read_parquet(carpeta + "vehiculos.parquet").rename(columns={
        "cod_ine": "codigo_ine",
        "parque_total": "vehiculos",
    })

    p = pd.read_parquet(carpeta + "puntos_recarga.parquet").rename(columns={
        "refill_points_50kw": "puntos_rapidos",
    })

    m = m[m["poblacion"] > 10000].copy()

    if "sin_especificar" in v:
        v = v[v["sin_especificar"].eq(False)].copy()

    v["vehiculos_cero_eco"] = v["dist_0"] + v["dist_eco"]
    p["codigo_ine"] = p["NATCODE"].astype("string").str[-5:]

    for nombre, df in [("municipios", m), ("vehículos", v), ("recarga", p)]:
        df["codigo_ine"] = df["codigo_ine"].astype("string").str.strip().str.zfill(5)

        if df["codigo_ine"].isna().any() or df["codigo_ine"].duplicated().any():
            raise ValueError(f"Códigos INE nulos o duplicados en {nombre}")

    datos = m.merge(
        v[["codigo_ine", "vehiculos", "vehiculos_cero_eco"]],
        on="codigo_ine",
        how="left",
        validate="one_to_one",
    )

    return datos.merge(
        p[["codigo_ine", "puntos_rapidos"]],
        on="codigo_ine",
        how="left",
        validate="one_to_one",
    )


def generar_gold(carpeta, salida=None, pesos=None):
    if pesos is None:
        raise ValueError("Debes proporcionar los pesos para generar la tabla Gold.")

    d = cargar_datos(carpeta)

    pob = d["poblacion"].replace(0, np.nan)
    veh = d["vehiculos"].replace(0, np.nan)
    pob_anterior = d["poblacion_5y"].replace(0, np.nan)

    d["vehiculos_por_habitante"] = veh / pob
    d["cero_eco_por_1000"] = 1000 * d["vehiculos_cero_eco"] / pob
    d["puntos_rapidos_por_10000"] = 10000 * d["puntos_rapidos"] / veh
    d["crecimiento_poblacion"] = 100 * (pob / pob_anterior - 1)

    d[INDICADORES] = d[INDICADORES].replace([np.inf, -np.inf], np.nan)

    if d[INDICADORES].isna().any().any():
        raise ValueError(
            "Hay indicadores sin datos:\n"
            + d[INDICADORES].isna().sum().to_string()
        )

    # Normalización robusta P5-P95.
    for indicador in INDICADORES:
        p5, p95 = d[indicador].quantile([0.05, 0.95])

        if p95 > p5:
            valores = (d[indicador].clip(p5, p95) - p5) / (p95 - p5)
        else:
            valores = pd.Series(0.0, index=d.index)

        # Menos infraestructura relativa implica mayor oportunidad.
        if indicador == "puntos_rapidos_por_10000" and p95 > p5:
            valores = 1 - valores

        d[indicador + "_norm"] = valores

    d["opportunity_score"] = calcular_score(d, pesos)

    d = d.sort_values(
        ["opportunity_score", "codigo_ine"],
        ascending=[False, True],
    ).reset_index(drop=True)

    d["ranking_general"] = np.arange(1, len(d) + 1)

    if salida is not None:
        Path(salida).parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(salida, index=False)

    return d


def calcular_score(datos, pesos):
    if len(pesos) != 4 or any(p < 0 for p in pesos) or not np.isclose(sum(pesos), 1):
        raise ValueError("Los cuatro pesos deben ser positivos o cero y sumar 1")

    return 100 * sum(
        datos[indicador + "_norm"] * peso
        for indicador, peso in zip(INDICADORES, pesos)
    )


def seleccionar_top20(datos, criterio="opportunity_score"):
    orden = datos.sort_values(
        [criterio, "codigo_ine"],
        ascending=[False, True],
    ).copy()

    orden["numero_provincia"] = orden.groupby("provincia").cumcount() + 1

    return (
        orden[orden["numero_provincia"] <= 3]
        .head(20)
        .drop(columns="numero_provincia")
        .reset_index(drop=True)
    )


# ============================================================
# 2. CALIDAD Y ANÁLISIS EXPLORATORIO
# ============================================================

def comprobar_gold(gold):
    seccion("CALIDAD DE LA TABLA GOLD")

    print(f"Municipios: {len(gold)}")
    print(f"Códigos INE duplicados: {gold.codigo_ine.duplicated().sum()}")
    print("\nValores faltantes:")
    print(gold.isna().sum().to_string())

    if (
        not gold.codigo_ine.is_unique
        or gold[INDICADORES].isna().any().any()
        or not gold.opportunity_score.between(0, 100).all()
    ):
        raise ValueError("La tabla Gold no supera las comprobaciones")

    seccion("ESTADÍSTICAS DE LOS INDICADORES")

    tabla(
        gold[INDICADORES]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T.reset_index()
    )

    print(
        f"\nPernoctaciones sin dato: "
        f"{gold['pernoctaciones_2025'].isna().sum()}/{len(gold)}"
    )
    print("No se imputan ni se incluyen en el score.")


def analizar_atipicos(gold):
    seccion("VALORES ATÍPICOS (IQR)")

    for indicador, etiqueta in zip(INDICADORES, ETIQUETAS):
        q1, q3 = gold[indicador].quantile([0.25, 0.75])
        iqr = q3 - q1
        minimo = q1 - 1.5 * iqr
        maximo = q3 + 1.5 * iqr

        n = ((gold[indicador] < minimo) | (gold[indicador] > maximo)).sum()

        print(f"{etiqueta}: {n} atípicos")


def analizar_posibles_flotas(gold):
    seccion("POSIBLES FLOTAS: REVISIÓN PENDIENTE")

    sospechosos = gold[
        (gold["vehiculos_por_habitante"] > 2)
        | (gold["cero_eco_por_1000"] > 200)
    ].copy()

    top20 = seleccionar_top20(gold)
    puestos = dict(zip(top20.codigo_ine, range(1, 21)))

    sospechosos["puesto_top20"] = (
        sospechosos["codigo_ine"].map(puestos).fillna("-")
    )

    print(f"Municipios detectados: {len(sospechosos)}")

    tabla(sospechosos, [
        "municipio", "provincia",
        "vehiculos_por_habitante",
        "cero_eco_por_1000",
        "puesto_top20",
    ])

    print("No se excluyen municipios sin validación de negocio.")


# ============================================================
# 3. RANKING Y BASELINE
# ============================================================

def comparar_con_baseline(gold):
    modelo = seleccionar_top20(gold)
    baseline = seleccionar_top20(gold, "poblacion")

    columnas = ["municipio", "provincia", "poblacion", "opportunity_score"]

    seccion("TOP 20: OPPORTUNITY SCORE")
    tabla(modelo, columnas)

    seccion("TOP 20: BASELINE POR POBLACIÓN")
    tabla(baseline, columnas)

    comunes = set(modelo.codigo_ine) & set(baseline.codigo_ine)

    seccion("COMPARACIÓN MODELO VS BASELINE")
    print(f"Coincidencias: {len(comunes)}/20")
    print(f"Score medio modelo: {modelo.opportunity_score.mean():.2f}")
    print(f"Score medio baseline: {baseline.opportunity_score.mean():.2f}")

    return modelo, baseline


# ============================================================
# 4. ANÁLISIS DE SENSIBILIDAD
# ============================================================

def analizar_sensibilidad(gold, escenarios, escenario_referencia=None):
    if not escenarios:
        raise ValueError("Debes proporcionar al menos un escenario de pesos.")
    if escenario_referencia is None:
        escenario_referencia = next(iter(escenarios))
    if escenario_referencia not in escenarios:
        raise ValueError(
            f"El escenario de referencia '{escenario_referencia}' no existe."
        )

    resultados = gold[["codigo_ine", "municipio", "provincia"]].copy()

    for nombre, pesos in escenarios.items():
        ranking = gold.copy()
        ranking["score_escenario"] = calcular_score(ranking, pesos)

        top20 = seleccionar_top20(ranking, "score_escenario")

        resultados[nombre] = resultados.codigo_ine.isin(top20.codigo_ine)

    nombres_escenarios = list(escenarios)
    resultados["apariciones"] = resultados[nombres_escenarios].sum(axis=1)

    resultados = (
        resultados[resultados["apariciones"] > 0]
        .sort_values(
            ["apariciones", escenario_referencia, "municipio"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    seccion("ANÁLISIS DE SENSIBILIDAD")

    print("Escenarios y pesos:")
    for nombre, pesos in escenarios.items():
        print(f"{nombre}: {[round(p * 100, 1) for p in pesos]} %")

    tabla(resultados, [
        "municipio", "provincia", "apariciones", *nombres_escenarios
    ])

    estables = (resultados["apariciones"] == len(escenarios)).sum()

    print(
        f"\nMunicipios presentes en todos los {len(escenarios)} escenarios: "
        f"{estables}"
    )
    print(f"Municipios presentes en un único escenario: "
          f"{(resultados['apariciones'] == 1).sum()}")

    return resultados


# ============================================================
# 5. K-MEANS
# ============================================================

def analizar_kmeans(gold):
    columnas = [c + "_norm" for c in INDICADORES]
    x = gold[columnas].to_numpy()

    metricas = []

    for k in range(2, 9):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        grupos = modelo.fit_predict(x)

        metricas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silhouette_score(x, grupos),
        })

    metricas = pd.DataFrame(metricas)
    k = int(metricas.loc[metricas.silueta.idxmax(), "k"])

    resultado = gold.copy()
    resultado["cluster"] = (
        KMeans(n_clusters=k, random_state=42, n_init=10)
        .fit_predict(x) + 1
    )

    seccion("K-MEANS: ELECCIÓN DE K")
    tabla(metricas)
    print(f"\nK seleccionado: {k}")

    seccion("PERFILES DE LOS CLÚSTERES")

    perfiles = resultado.groupby("cluster").agg(
        municipios=("codigo_ine", "size"),
        score_medio=("opportunity_score", "mean"),
        **{c: (c, "mean") for c in INDICADORES},
    ).reset_index()

    tabla(perfiles)

    seccion("CLÚSTERES DEL TOP 20")
    tabla(
        seleccionar_top20(resultado),
        ["municipio", "provincia", "opportunity_score", "cluster"],
    )

    return resultado, metricas


# ============================================================
# 6. VISOR DE GRÁFICOS
# ============================================================

def visor(titulo, paginas, ancho=12, alto=7):
    fig = plt.figure(figsize=(ancho, alto))
    estado = {"pagina": 0}

    anterior_ax = fig.add_axes([0.32, 0.02, 0.15, 0.055])
    siguiente_ax = fig.add_axes([0.53, 0.02, 0.15, 0.055])

    anterior = Button(anterior_ax, "Anterior")
    siguiente = Button(siguiente_ax, "Siguiente")

    def dibujar():
        for ax in list(fig.axes):
            if ax not in (anterior_ax, siguiente_ax):
                ax.remove()

        for texto in list(fig.texts):
            texto.remove()

        paginas[estado["pagina"]](fig)

        fig.text(
            0.5, 0.105,
            f"{titulo}  |  {estado['pagina'] + 1}/{len(paginas)}",
            ha="center",
        )

        fig.canvas.draw_idle()

    def mover(paso):
        estado["pagina"] = (estado["pagina"] + paso) % len(paginas)
        dibujar()

    anterior.on_clicked(lambda event: mover(-1))
    siguiente.on_clicked(lambda event: mover(1))

    def teclado(event):
        if event.key == "left":
            mover(-1)
        elif event.key == "right":
            mover(1)

    fig.canvas.mpl_connect("key_press_event", teclado)
    fig._botones = (anterior, siguiente)

    dibujar()
    return fig


# ============================================================
# 7. GRÁFICOS DE INDICADORES
# ============================================================

def graficos_indicadores(gold):
    paginas = []

    for indicador, etiqueta in zip(INDICADORES, ETIQUETAS):

        def pagina(fig, col=indicador, nombre=etiqueta):
            valores = gold[col].dropna()
            p5, mediana, p95 = valores.quantile([0.05, 0.5, 0.95])

            # Histograma y boxplot juntos.
            ax1 = fig.add_axes([0.10, 0.27, 0.48, 0.55])
            ax2 = fig.add_axes([0.67, 0.27, 0.25, 0.55])

            ax1.hist(
                valores, bins=25,
                color="#5b9c98", edgecolor="white",
            )

            for valor, texto, color in [
                (p5, "P5", "#bd7d22"),
                (mediana, "Mediana", "black"),
                (p95, "P95", "#bf4e38"),
            ]:
                ax1.axvline(
                    valor, color=color, linestyle="--",
                    label=f"{texto}: {valor:.2f}",
                )

            ax1.set_title("Histograma")
            ax1.set_xlabel(nombre)
            ax1.set_ylabel("Municipios")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.2)

            ax2.boxplot(
                valores,
                patch_artist=True,
                boxprops={"facecolor": "#a6d4cd"},
                medianprops={"color": "black", "linewidth": 2},
            )

            ax2.set_title("Boxplot")
            ax2.set_ylabel(nombre)
            ax2.grid(alpha=0.2)

            fig.suptitle(nombre, fontsize=15, y=0.93)

            fig.text(
                0.5, 0.17,
                "Histograma y boxplot: todos los valores. El score limita extremos a P5-P95.",
                ha="center", fontsize=9,
            )

        paginas.append(pagina)

    def correlacion(fig):
        matriz = gold[INDICADORES].corr()
        nombres = ["Veh/hab.", "CERO/ECO", "Recarga", "Crecimiento"]

        ax = fig.add_axes([0.20, 0.22, 0.60, 0.63])
        imagen = ax.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks(range(4), nombres, rotation=20)
        ax.set_yticks(range(4), nombres)

        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, f"{matriz.iloc[i, j]:.2f}",
                    ha="center", va="center",
                )

        fig.colorbar(imagen, ax=ax)
        ax.set_title("Correlación de Pearson")

    paginas.append(correlacion)

    return visor("Indicadores", paginas)


# ============================================================
# 8. GRÁFICOS DE RANKING
# ============================================================


def grafico_comparacion(modelo, baseline):
    paginas = []

    for datos, titulo, criterio, color in [
        (baseline, "Baseline por población", "poblacion", "#678bb3"),
        (modelo, "Modelo por Opportunity Score", "opportunity_score", "#438a86"),
    ]:

        def pagina(fig, df=datos, nombre=titulo, columna=criterio, tono=color):
            ax = fig.add_axes([0.38, 0.16, 0.55, 0.73])

            orden = df.sort_values(columna, ascending=True)

            etiquetas = [
                f"{fila.municipio} ({fila.provincia})"
                for _, fila in orden.iterrows()
            ]

            barras = ax.barh(etiquetas, orden[columna], color=tono)

            for barra, (_, fila) in zip(barras, orden.iterrows()):
                valor = fila[columna]

                texto = (
                    f"{valor:,.0f} hab. | Score: {fila.opportunity_score:.1f}"
                    if columna == "poblacion"
                    else f"{valor:.1f}"
                )

                ax.text(
                    valor,
                    barra.get_y() + barra.get_height() / 2,
                    f"  {texto}",
                    va="center",
                    fontsize=8,
                )

            ax.set_xlim(0, orden[columna].max() * 1.48)
            ax.set_xlabel(
                "Población (habitantes)"
                if columna == "poblacion"
                else "Opportunity Score (0-100)"
            )
            ax.set_title(nombre)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

        paginas.append(pagina)

    return visor("Comparación de rankings", paginas, 15, 9)


# ============================================================
# 9. GRÁFICOS DE K-MEANS
# ============================================================

def graficos_kmeans(resultado, metricas):
    paginas = []
    columnas = [c + "_norm" for c in INDICADORES]
    k = resultado.cluster.nunique()

    def codo(fig):
        ax = fig.add_axes([0.15, 0.25, 0.72, 0.58])

        ax.plot(metricas.k, metricas.inercia, marker="o")
        ax.axvline(k, color="tomato", linestyle="--", label=f"K = {k}")

        ax.set_xticks(metricas.k)
        ax.set_xlabel("Número de clústeres")
        ax.set_ylabel("Inercia")
        ax.set_title("Método del codo")
        ax.legend()
        ax.grid(alpha=0.2)

    paginas.append(codo)

    def silueta(fig):
        ax = fig.add_axes([0.15, 0.25, 0.72, 0.58])

        ax.plot(metricas.k, metricas.silueta, marker="o")
        ax.axvline(k, color="tomato", linestyle="--", label=f"K = {k}")

        ax.set_xticks(metricas.k)
        ax.set_xlabel("Número de clústeres")
        ax.set_ylabel("Coeficiente de silueta")
        ax.set_title("Método de la silueta")
        ax.legend()
        ax.grid(alpha=0.2)

    paginas.append(silueta)

    def perfiles(fig):
        ax = fig.add_axes([0.10, 0.26, 0.83, 0.58])
        medias = resultado.groupby("cluster")[columnas].mean()

        for grupo, fila in medias.iterrows():
            ax.plot(
                range(4), fila.to_numpy(),
                marker="o", label=f"Grupo {grupo}",
            )

        ax.set_xticks(
            range(4),
            ["Veh/hab.", "CERO/ECO", "Escasez recarga", "Crecimiento"],
        )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Media normalizada")
        ax.set_title("Perfiles de los clústeres")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(alpha=0.2)

        fig.text(
            0.5, 0.17,
            "Escasez alta = menor disponibilidad relativa de recarga rápida.",
            ha="center", fontsize=9,
        )

    paginas.append(perfiles)

    def pca(fig):
        x = resultado[columnas].to_numpy()
        modelo = PCA(n_components=2)
        puntos = modelo.fit_transform(x)

        ax = fig.add_axes([0.13, 0.23, 0.73, 0.62])

        for grupo in sorted(resultado.cluster.unique()):
            mascara = resultado.cluster.eq(grupo).to_numpy()

            ax.scatter(
                puntos[mascara, 0],
                puntos[mascara, 1],
                s=22, alpha=0.65,
                label=f"Grupo {grupo}",
            )

        ax.set_xlabel(
            f"Componente 1 ({modelo.explained_variance_ratio_[0]:.1%})"
        )
        ax.set_ylabel(
            f"Componente 2 ({modelo.explained_variance_ratio_[1]:.1%})"
        )

        ax.set_title("Representación PCA")
        ax.legend()
        ax.grid(alpha=0.2)

    paginas.append(pca)

    return visor("K-Means", paginas)
