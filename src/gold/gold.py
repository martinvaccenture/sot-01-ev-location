
"""Tabla Gold y análisis del proyecto EV Location Intelligence."""

from pathlib import Path
import unicodedata

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
    carpeta = Path(carpeta)

    m = pd.read_parquet(carpeta / "municipios.parquet").rename(columns={
        "cod_ine": "codigo_ine", "nombre": "municipio",
        "pob_2025": "poblacion", "pob_2020": "poblacion_5y",
    })

    v = pd.read_parquet(carpeta / "vehiculos.parquet").rename(columns={
        "cod_ine": "codigo_ine", "parque_total": "vehiculos",
    })

    p = pd.read_parquet(carpeta / "puntos_recarga.parquet").rename(columns={
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
    orden = datos.sort_values([criterio, "codigo_ine"], ascending=[False, True]).copy()
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
    sospechosos["puesto_top20"] = sospechosos["codigo_ine"].map(puestos).fillna("-")
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
    """Resume la frecuencia de selección y los cambios entre los escenarios."""
    gold = preparacion["gold"]
    top20_escenarios = preparacion["top20_escenarios"]
    nombres = preparacion["nombres_escenarios"]
    apariciones = preparacion["apariciones"]
    n_escenarios = len(top20_escenarios)
    referencia = preparacion["escenario_referencia"]
    referencia_key = (referencia, False)
    codigos_referencia = set(top20_escenarios[referencia_key]["codigo_ine"])

    seccion("ANÁLISIS DE SENSIBILIDAD")
    print(f"Escenarios evaluados: {n_escenarios}")
    print(
        "Pernoctaciones disponibles: "
        f"{preparacion['n_municipios_turismo']}/{len(gold)} municipios"
    )
    print("Peso turístico aplicado solo con dato válido:")
    print(f"{preparacion['peso_turismo']:.1%}")
    print(
        "Aparición: número de Top 20 en los que entra el municipio "
        "dividido por el total de escenarios."
    )

    print("\nPesos efectivos por escenario:")
    for clave, pesos in preparacion["pesos_escenarios"].items():
        print(f"{nombres[clave]}: {[round(p * 100, 1) for p in pesos]} %")

    columnas_escenarios = [nombres[clave] for clave in top20_escenarios]
    tabla_apariciones = apariciones[
        apariciones["aparicion_escenarios"] > 0
    ].sort_values(
        ["aparicion_escenarios", "municipio"],
        ascending=[False, True],
    )
    tabla(tabla_apariciones[
        ["municipio", "provincia", "aparicion_escenarios", *columnas_escenarios]
    ])

    n_estables = int(np.isclose(apariciones["aparicion_escenarios"], 1).sum())
    n_unicos = int(np.isclose(
        apariciones["aparicion_escenarios"], 1 / n_escenarios
    ).sum())
    print(f"\nMunicipios presentes en todos los {n_escenarios} escenarios: {n_estables}")
    print(f"Municipios presentes en un único escenario: {n_unicos}")

    seccion("CAMBIOS RESPECTO A ACTUAL SIN TURISMO")
    for clave, top in top20_escenarios.items():
        if clave == referencia_key:
            continue
        codigos_nuevos = set(top["codigo_ine"])
        print(
            f"{nombres[clave]}: entran {len(codigos_nuevos - codigos_referencia)}, "
            f"salen {len(codigos_referencia - codigos_nuevos)}"
        )

    seccion("MUNICIPIOS ESTABLES Y SUS CARACTERÍSTICAS")
    estables = gold.loc[
        np.isclose(gold["aparicion_escenarios"], 1)
    ].sort_values("ranking_general")
    columnas_estables = [
        "municipio", "provincia", *INDICADORES,
        "opportunity_score", "ranking_general", "aparicion_escenarios",
    ]
    tabla(estables[columnas_estables])
    print("\nLos indicadores son valores originales sin normalizar.")
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


def _slug_escenario(nombre):
    nombre_ascii = unicodedata.normalize("NFKD", nombre)
    nombre_ascii = nombre_ascii.encode("ascii", "ignore").decode("ascii")
    return "_".join(nombre_ascii.lower().split())


def preparar_escenarios(
    gold,
    escenarios,
    peso_turismo,
    normalizacion,
    umbrales_flotas,
    escenario_referencia="Actual",
):
    """Prepara diez rankings, su frecuencia y las cuatro tablas de referencia.

    Los escenarios turísticos sustituyen únicamente el peso reservado a turismo
    cuando existe el dato. Si falta, se conserva íntegro el score de cuatro
    indicadores; no se imputa ni se penaliza al municipio.
    """
    if not escenarios:
        raise ValueError("Debes proporcionar escenarios base de pesos.")
    if escenario_referencia not in escenarios:
        raise ValueError(
            f"El escenario de referencia '{escenario_referencia}' no existe."
        )
    if not 0 <= peso_turismo < 1:
        raise ValueError("El peso de turismo debe estar entre 0 y 1, sin incluir 1.")

    pesos_turismo = normalizar_pernoctaciones(gold, normalizacion)
    riesgo_flotas = identificar_posibles_flotas(gold, umbrales_flotas)
    rankings = {}
    top20_escenarios = {}
    nombres_escenarios = {}
    pesos_escenarios = {}
    gold_global = gold.copy()
    gold_global["posible_riesgo_flota"] = riesgo_flotas

    for nombre, pesos in escenarios.items():
        score_base = calcular_score(gold, pesos)

        for incluir_turismo in (False, True):
            clave = (nombre, incluir_turismo)
            sufijo = "con_turismo" if incluir_turismo else "sin_turismo"
            nombre_completo = f"{nombre} ({sufijo.replace('_', ' ')})"
            slug = f"{_slug_escenario(nombre)}_{sufijo}"
            score = score_base.copy()

            if incluir_turismo:
                tiene_turismo = pesos_turismo.notna()
                score.loc[tiene_turismo] = (
                    (1 - peso_turismo) * score_base.loc[tiene_turismo]
                    + 100 * peso_turismo * pesos_turismo.loc[tiene_turismo]
                )
                pesos_efectivos = [
                    peso * (1 - peso_turismo) for peso in pesos
                ] + [peso_turismo]
            else:
                pesos_efectivos = list(pesos)

            ranking = gold.copy()
            ranking["score_escenario"] = score
            ranking["posible_riesgo_flota"] = riesgo_flotas
            ranking = ranking.sort_values(
                ["score_escenario", "codigo_ine"],
                ascending=[False, True],
            ).reset_index(drop=True)
            ranking["ranking_general_escenario"] = np.arange(
                1, len(ranking) + 1
            )

            top20 = seleccionar_top20(ranking, "score_escenario")
            top20["posicion_top20"] = np.arange(1, len(top20) + 1)
            top20["nombre_escenario"] = nombre_completo

            rankings[clave] = ranking
            top20_escenarios[clave] = top20
            nombres_escenarios[clave] = nombre_completo
            pesos_escenarios[clave] = pesos_efectivos

            indice = ranking.set_index("codigo_ine")
            puestos_top20 = top20.set_index("codigo_ine")["posicion_top20"]
            gold_global[f"score_{slug}"] = gold_global["codigo_ine"].map(
                indice["score_escenario"]
            )
            gold_global[f"ranking_general_{slug}"] = gold_global[
                "codigo_ine"
            ].map(indice["ranking_general_escenario"])
            gold_global[f"top20_{slug}"] = gold_global["codigo_ine"].map(
                puestos_top20
            ).astype("Int64")

    apariciones = pd.Series(0, index=gold_global.index, dtype="int64")
    for top20 in top20_escenarios.values():
        apariciones += gold_global["codigo_ine"].isin(
            top20["codigo_ine"]
        ).astype("int64")

    gold_global["aparicion_escenarios"] = apariciones / len(top20_escenarios)
    tabla_apariciones = gold_global[[
        "codigo_ine", "municipio", "provincia", "aparicion_escenarios",
    ]].copy()
    for clave, top20 in top20_escenarios.items():
        tabla_apariciones[nombres_escenarios[clave]] = tabla_apariciones[
            "codigo_ine"
        ].isin(top20["codigo_ine"])

    for ranking in rankings.values():
        ranking["aparicion_escenarios"] = ranking["codigo_ine"].map(
            gold_global.set_index("codigo_ine")["aparicion_escenarios"]
        )
    for top20 in top20_escenarios.values():
        top20["aparicion_escenarios"] = top20["codigo_ine"].map(
            gold_global.set_index("codigo_ine")["aparicion_escenarios"]
        )

    tablas_finales = {}
    variantes = {
        "sin_turismo": (False, False),
        "con_turismo": (True, False),
        "sin_turismo_sin_flotas": (False, True),
        "con_turismo_sin_flotas": (True, True),
    }
    columnas_finales = [
        "codigo_ine", "municipio", "provincia", "poblacion",
        "pernoctaciones_2025", "opportunity_score",
        "ranking_general", "posicion_top20", "aparicion_escenarios",
        "posible_riesgo_flota",
    ]

    for nombre_tabla, (incluir_turismo, excluir_flotas) in variantes.items():
        ranking = rankings[(escenario_referencia, incluir_turismo)].copy()
        if excluir_flotas:
            ranking = ranking.loc[~ranking["posible_riesgo_flota"]]

        seleccion = seleccionar_top20(ranking, "score_escenario").copy()
        seleccion["opportunity_score"] = seleccion["score_escenario"]
        seleccion["ranking_general"] = seleccion["ranking_general_escenario"]
        seleccion["posicion_top20"] = np.arange(1, len(seleccion) + 1)
        tablas_finales[nombre_tabla] = seleccion[columnas_finales].copy()

    return {
        "gold": gold_global,
        "rankings_escenarios": rankings,
        "top20_escenarios": top20_escenarios,
        "nombres_escenarios": nombres_escenarios,
        "pesos_escenarios": pesos_escenarios,
        "apariciones": tabla_apariciones,
        "tablas_finales": tablas_finales,
        "n_municipios_turismo": int(pesos_turismo.notna().sum()),
        "peso_turismo": peso_turismo,
        "escenario_referencia": escenario_referencia,
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
