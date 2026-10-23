
"""Tabla Gold y análisis del proyecto EV Location Intelligence."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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
    # String y no Path: Path rompe las rutas S3 ("s3://" -> "s3:/").
    carpeta = str(carpeta)
    if not carpeta.endswith(("/", "\\")):
        carpeta += "/"

    m = pd.read_parquet(carpeta + "municipios.parquet").rename(columns={
        "cod_ine": "codigo_ine", "nombre": "municipio",
        "pob_2025": "poblacion", "pob_2020": "poblacion_5y",
    })

    v = pd.read_parquet(carpeta + "vehiculos.parquet").rename(columns={
        "cod_ine": "codigo_ine", "parque_total": "vehiculos",
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

    columnas_vehiculos = ["codigo_ine", "vehiculos", "vehiculos_cero_eco"]
    if "riesgo_flota" in v:
        columnas_vehiculos.append("riesgo_flota")

    datos = m.merge(
        v[columnas_vehiculos],
        on="codigo_ine", how="left", validate="one_to_one",
    )

    return datos.merge(
        p[["codigo_ine", "puntos_rapidos"]],
        on="codigo_ine", how="left", validate="one_to_one",
    )


def generar_gold(carpeta, salida=None, pesos=None, normalizacion="p5_p95"):
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
        raise ValueError("Hay indicadores sin datos:\n" + d[INDICADORES].isna().sum().to_string())

    if normalizacion not in ("p5_p95", "minmax"):
        raise ValueError("La normalización debe ser 'p5_p95' o 'minmax'.")

    for indicador in INDICADORES:
        if normalizacion == "p5_p95":
            minimo, maximo = d[indicador].quantile([0.05, 0.95])
            datos_normalizar = d[indicador].clip(minimo, maximo)
        else:
            minimo, maximo = d[indicador].min(), d[indicador].max()
            datos_normalizar = d[indicador]

        if maximo > minimo:
            valores = (datos_normalizar - minimo) / (maximo - minimo)
        else:
            valores = pd.Series(0.0, index=d.index)

        if indicador == "puntos_rapidos_por_10000" and maximo > minimo:
            valores = 1 - valores

        d[indicador + "_norm"] = valores

    d["opportunity_score"] = calcular_score(d, pesos)
    d = d.sort_values(["opportunity_score", "codigo_ine"], ascending=[False, True]).reset_index(drop=True)
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


def seleccionar_top20(datos, criterio="opportunity_score", n=20):
    criterios = [criterio] if isinstance(criterio, str) else list(criterio)
    orden = datos.sort_values(
        [*criterios, "codigo_ine"],
        ascending=[False] * len(criterios) + [True],
    ).copy()
    orden["numero_provincia"] = orden.groupby("provincia").cumcount() + 1

    return (
        orden[orden["numero_provincia"] <= 3]
        .head(n)
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

    if (not gold.codigo_ine.is_unique or gold[INDICADORES].isna().any().any()
            or not gold.opportunity_score.between(0, 100).all()):
        raise ValueError("La tabla Gold no supera las comprobaciones")

    seccion("ESTADÍSTICAS DE LOS INDICADORES")
    tabla(gold[INDICADORES].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index())

    print(f"\nPernoctaciones sin dato: {gold['pernoctaciones_2025'].isna().sum()}/{len(gold)}")
    print("No se imputan ni se incluyen en el score inicial.")


def analizar_atipicos(gold):
    seccion("VALORES ATÍPICOS (IQR)")

    for indicador, etiqueta in zip(INDICADORES, ETIQUETAS):
        q1, q3 = gold[indicador].quantile([0.25, 0.75])
        iqr = q3 - q1
        minimo, maximo = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((gold[indicador] < minimo) | (gold[indicador] > maximo)).sum()
        print(f"{etiqueta}: {n} atípicos")


def identificar_posibles_flotas(gold, umbrales_flotas):
    columnas_requeridas = {
        "vehiculos_por_habitante",
        "cero_eco_por_1000",
    }
    if set(umbrales_flotas) != columnas_requeridas:
        raise ValueError(
            "Los umbrales deben definir vehículos por habitante y CERO/ECO por 1.000."
        )

    return (
        gold["vehiculos_por_habitante"].gt(
            umbrales_flotas["vehiculos_por_habitante"]
        )
        | gold["cero_eco_por_1000"].gt(umbrales_flotas["cero_eco_por_1000"])
    )


def analizar_posibles_flotas(gold, umbrales_flotas):
    seccion("POSIBLES FLOTAS: REVISIÓN PENDIENTE")

    sospechosos = gold[identificar_posibles_flotas(gold, umbrales_flotas)].copy()

    top20 = seleccionar_top20(gold)
    puestos = dict(zip(top20.codigo_ine, range(1, 21)))
    sospechosos["puesto_top20"] = sospechosos["codigo_ine"].map(puestos).astype("string").fillna("-")
    sospechosos = sospechosos.sort_values("ranking_general")

    print(f"Municipios detectados: {len(sospechosos)}")
    print("Ranking general: posición entre los municipios, sin límite provincial.")
    print("Puesto Top 20: posición con máximo 3 por provincia.")

    tabla(sospechosos, [
        "municipio", "provincia", "vehiculos_por_habitante",
        "cero_eco_por_1000", "ranking_general", "puesto_top20",
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

def analizar_sensibilidad(preparacion):
    """Resume pesos, apariciones en los Top 20 y cambios entre escenarios."""
    resumen = preparacion["gold"]
    tablas = preparacion["tablas_escenarios"]
    claves = list(tablas)
    referencia = claves[0]

    seccion("ANÁLISIS DE SENSIBILIDAD")
    print(f"Escenarios evaluados: {len(claves)}")
    print(
        "Municipios con pernoctaciones: "
        f"{preparacion['n_municipios_turismo']}/{len(resumen)}"
    )
    print(f"Origen del riesgo de flota: {preparacion['origen_riesgo']}")
    print(resumen["riesgo_flota"].value_counts().to_string())

    print("\nPesos efectivos (vehículos, CERO/ECO, recarga, crecimiento, turismo):")
    for clave, pesos in preparacion["pesos_escenarios"].items():
        print(f"{clave}: {[round(p * 100, 1) for p in pesos]} %")

    seleccionados = resumen[resumen["apariciones"] > 0].sort_values(
        ["aparicion_escenarios", "score_medio_total", "codigo_ine"],
        ascending=[False, False, True],
    )
    tabla(seleccionados, [
        "municipio", "provincia", "riesgo_flota", "apariciones",
        "aparicion_escenarios", "score_medio_total",
    ])
    print(f"\nPresentes en los {len(claves)} escenarios: "
          f"{(resumen['apariciones'] == len(claves)).sum()}")
    print(f"Presentes en un único escenario: {(resumen['apariciones'] == 1).sum()}")

    seccion(f"CAMBIOS RESPECTO A {referencia}")
    en_top = lambda t: set(t.loc[t["ranking_20"].notna(), "codigo_ine"])
    codigos_referencia = en_top(tablas[referencia])
    for clave in claves[1:]:
        codigos = en_top(tablas[clave])
        print(f"{clave}: entran {len(codigos - codigos_referencia)}, "
              f"salen {len(codigos_referencia - codigos)}")

    seccion("MUNICIPIOS PRESENTES EN TODOS LOS ESCENARIOS")
    tabla(
        resumen[resumen["apariciones"] == len(claves)],
        ["municipio", "provincia", *INDICADORES, "score_medio"],
    )
    print("\nIndicadores originales sin normalizar.")
    return preparacion


# ============================================================
# 5. TURISMO Y ESCENARIOS
# ============================================================

def normalizar_pernoctaciones(gold, normalizacion="p5_p95"):
    """Normaliza el turismo solo entre los municipios con dato disponible."""
    if normalizacion not in ("p5_p95", "minmax"):
        raise ValueError("La normalización debe ser 'p5_p95' o 'minmax'.")

    valores = pd.to_numeric(gold["pernoctaciones_2025"], errors="coerce")
    validos = valores.dropna()
    if validos.empty:
        return pd.Series(np.nan, index=gold.index)

    if normalizacion == "p5_p95":
        minimo, maximo = validos.quantile([0.05, 0.95])
        valores_normalizar = valores.clip(minimo, maximo)
    else:
        minimo, maximo = validos.min(), validos.max()
        valores_normalizar = valores

    if maximo == minimo:
        return pd.Series(0.0, index=gold.index).where(valores.notna())

    return ((valores_normalizar - minimo) / (maximo - minimo)).clip(0, 1)


def clasificar_riesgo_flota(gold, umbrales_flotas):
    """Usa riesgo_flota de vehiculos.parquet; si aún no existe, umbrales provisionales."""
    if "riesgo_flota" in gold and gold["riesgo_flota"].notna().any():
        riesgo = (
            gold["riesgo_flota"].astype("string").str.strip().str.lower()
            .replace("", pd.NA).fillna("desconocido")
        )
        return riesgo, "vehiculos.parquet"

    sospechoso = identificar_posibles_flotas(gold, umbrales_flotas)
    riesgo = pd.Series(
        np.where(sospechoso, "posible riesgo", "normal"), index=gold.index
    )
    return riesgo, "umbrales provisionales (sin columna riesgo_flota)"


def _contar_apariciones(rankings, codigos, mascara=None):
    """Cuenta en cuántos Top 20 de escenario aparece cada municipio."""
    apariciones = pd.Series(0, index=codigos.index, dtype="int64")
    for ranking in rankings.values():
        candidatos = ranking if mascara is None else ranking[
            ranking["codigo_ine"].map(mascara)
        ]
        top20 = seleccionar_top20(candidatos, "score")
        apariciones += codigos.isin(top20["codigo_ine"]).astype("int64")
    return apariciones


def preparar_escenarios(
    gold,
    escenarios,
    normalizacion,
    umbrales_flotas,
    valores_riesgo_excluidos,
):
    """Genera los rankings por escenario (con y sin turismo) y los Top 20 agregados.

    escenarios: {nombre: {"pesos": [4 pesos], "peso_turismo": float}}.
    El turismo solo se aplica a municipios con pernoctaciones; sin dato se
    conserva el score de cuatro indicadores, sin imputar ni penalizar.
    """
    if not escenarios:
        raise ValueError("Debes proporcionar escenarios de pesos.")

    turismo_norm = normalizar_pernoctaciones(gold, normalizacion)
    tiene_turismo = turismo_norm.notna()
    riesgo, origen_riesgo = clasificar_riesgo_flota(gold, umbrales_flotas)

    base = gold.copy()
    base["riesgo_flota"] = riesgo
    columnas_tabla = [
        "codigo_ine", "municipio", "provincia", "poblacion",
        "pernoctaciones_2025", *INDICADORES, "riesgo_flota",
    ]

    tablas_escenarios = {}
    pesos_escenarios = {}

    for nombre, config in escenarios.items():
        pesos, peso_turismo = config["pesos"], config["peso_turismo"]
        if not 0 <= peso_turismo < 1:
            raise ValueError(f"Peso de turismo no válido en '{nombre}'.")

        score_base = calcular_score(base, pesos)
        score_turismo = score_base.copy()
        score_turismo[tiene_turismo] = (
            (1 - peso_turismo) * score_base[tiene_turismo]
            + 100 * peso_turismo * turismo_norm[tiene_turismo]
        )

        variantes = {
            nombre: (score_base, [*pesos, 0.0]),
            f"{nombre}_turismo": (
                score_turismo,
                [p * (1 - peso_turismo) for p in pesos] + [peso_turismo],
            ),
        }
        for clave, (score, pesos_efectivos) in variantes.items():
            ranking = base[columnas_tabla].assign(score=score).sort_values(
                ["score", "codigo_ine"], ascending=[False, True]
            ).reset_index(drop=True)
            ranking["ranking_general"] = np.arange(1, len(ranking) + 1)

            top20 = seleccionar_top20(ranking, "score")
            puestos = dict(zip(top20["codigo_ine"], range(1, len(top20) + 1)))
            ranking["ranking_20"] = ranking["codigo_ine"].map(puestos).astype("Int64")

            tablas_escenarios[clave] = ranking
            pesos_escenarios[clave] = pesos_efectivos

    resumen = base.copy()
    for clave, ranking in tablas_escenarios.items():
        indice = ranking.set_index("codigo_ine")
        resumen[f"score_{clave}"] = resumen["codigo_ine"].map(indice["score"])
        resumen[f"ranking_20_{clave}"] = resumen["codigo_ine"].map(
            indice["ranking_20"]
        ).astype("Int64")

    n = len(tablas_escenarios)
    claves_sin = [c for c in tablas_escenarios if not c.endswith("_turismo")]
    claves_con = [c for c in tablas_escenarios if c.endswith("_turismo")]
    resumen["score_medio"] = resumen[[f"score_{c}" for c in claves_sin]].mean(axis=1)
    resumen["score_medio_turismo"] = resumen[
        [f"score_{c}" for c in claves_con]
    ].mean(axis=1)
    resumen["score_medio_total"] = resumen[
        [f"score_{c}" for c in tablas_escenarios]
    ].mean(axis=1)

    resumen["apariciones"] = _contar_apariciones(tablas_escenarios, resumen["codigo_ine"])
    resumen["aparicion_escenarios"] = resumen["apariciones"] / n

    # Riesgo desconocido no se da por seguro hasta validar su clasificación.
    confiable = ~resumen["riesgo_flota"].isin(
        [*valores_riesgo_excluidos, "desconocido"]
    )

    columnas_top = [
        "codigo_ine", "municipio", "provincia", "poblacion",
        "pernoctaciones_2025", *INDICADORES, "riesgo_flota",
        "apariciones", "aparicion_escenarios", "score_medio", "score_medio_turismo",
    ]
    # Las apariciones son siempre las de los 8 escenarios originales, también en confianza.
    definiciones = {
        "top_20": (resumen, "score_medio"),
        "top_20_turismo": (resumen, "score_medio_turismo"),
        "top_20_confianza": (resumen[confiable], "score_medio"),
        "top_20_confianza_turismo": (resumen[confiable], "score_medio_turismo"),
    }
    tablas_top = {}
    for nombre, (datos, desempate) in definiciones.items():
        top = seleccionar_top20(datos, ["apariciones", desempate])[columnas_top].copy()
        top.insert(0, "ranking_20", np.arange(1, len(top) + 1))
        tablas_top[nombre] = top

    return {
        "gold": resumen,
        "tablas_escenarios": tablas_escenarios,
        "tablas_top": tablas_top,
        "pesos_escenarios": pesos_escenarios,
        "n_municipios_turismo": int(tiene_turismo.sum()),
        "origen_riesgo": origen_riesgo,
    }


# ============================================================
# 6. K-MEANS
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
        KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x) + 1
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
# 7. VISOR DE GRÁFICOS
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
            f"{titulo} | {estado['pagina'] + 1}/{len(paginas)}",
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
# 8. GRÁFICOS DE INDICADORES
# ============================================================

def graficos_indicadores(gold):
    paginas = []

    for indicador, etiqueta in zip(INDICADORES, ETIQUETAS):

        def pagina(fig, col=indicador, nombre=etiqueta):
            valores = gold[col].dropna()
            p5, mediana, p95 = valores.quantile([0.05, 0.5, 0.95])

            ax1 = fig.add_axes([0.10, 0.27, 0.48, 0.55])
            ax2 = fig.add_axes([0.67, 0.27, 0.25, 0.55])

            ax1.hist(valores, bins=25, color="#5b9c98", edgecolor="white")

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
                valores, patch_artist=True,
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
                ax.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center")

        fig.colorbar(imagen, ax=ax)
        ax.set_title("Correlación de Pearson")

    paginas.append(correlacion)
    return visor("Indicadores", paginas)


# ============================================================
# 9. GRÁFICOS DE RANKING
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
                    if columna == "poblacion" else f"{valor:.1f}"
                )
                ax.text(
                    valor, barra.get_y() + barra.get_height() / 2,
                    f"  {texto}", va="center", fontsize=8,
                )

            ax.set_xlim(0, orden[columna].max() * 1.48)
            ax.set_xlabel(
                "Población (habitantes)" if columna == "poblacion"
                else "Opportunity Score (0-100)"
            )
            ax.set_title(nombre)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

        paginas.append(pagina)

    return visor("Comparación de rankings", paginas, 15, 9)


# ============================================================
# 10. GRÁFICOS DE K-MEANS
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
            ax.plot(range(4), fila.to_numpy(), marker="o", label=f"Grupo {grupo}")

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
                puntos[mascara, 0], puntos[mascara, 1],
                s=22, alpha=0.65, label=f"Grupo {grupo}",
            )

        ax.set_xlabel(f"Componente 1 ({modelo.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"Componente 2 ({modelo.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Representación PCA")
        ax.legend()
        ax.grid(alpha=0.2)

    paginas.append(pca)
    return visor("K-Means", paginas)
