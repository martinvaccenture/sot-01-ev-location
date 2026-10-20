
"""EV Location Intelligence - Opportunity Score y análisis municipal.

Funciones para importar y ejecutar desde orchestrator.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Indicadores utilizados en el modelo
INDICADORES = [
    "vehiculos_por_habitante",
    "cero_eco_por_1000",
    "puntos_rapidos_por_10000",
    "crecimiento_poblacion",
    "turismo_por_habitante",
]

# Importancia de cada indicador en el Opportunity Score
PESOS = {
    "vehiculos_por_habitante": 0.15,
    "cero_eco_por_1000": 0.30,
    "puntos_rapidos_por_10000": 0.30,
    "crecimiento_poblacion": 0.15,
    "turismo_por_habitante": 0.10,
}

# Escenarios para el análisis de sensibilidad
ESCENARIOS = {
    "Base": PESOS,
    "CERO_ECO": dict(zip(INDICADORES, [0.15, 0.45, 0.20, 0.10, 0.10])),
    "Infraestructura": dict(zip(INDICADORES, [0.15, 0.20, 0.40, 0.10, 0.15])),
    "Turismo": dict(zip(INDICADORES, [0.15, 0.25, 0.20, 0.10, 0.30])),
}

COLUMNAS_NUMERICAS = [
    "poblacion",
    "poblacion_5y",
    "vehiculos",
    "vehiculos_cero_eco",
    "puntos_rapidos",
    "pernoctaciones",
]


def cargar_parquet(ruta):
    """Carga un archivo Parquet y devuelve su contenido como DataFrame."""
    return pd.read_parquet(ruta)


def mostrar_resumen_dataframe(df, nombre):
    """Muestra la estructura y una muestra legible de un DataFrame."""
    print(f"\n=== {nombre} ===")
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")

    print("\nColumnas y tipos:")
    print(df.dtypes.to_string())

    print("\nPrimeras filas:")
    muestra = df.drop(columns="geometry", errors="ignore").head()
    print(muestra.to_string(index=False))

    print("\nValores ausentes por columna:")
    print(df.isna().sum().to_string())


def diagnosticar_uniones(municipios, vehiculos, puntos_recarga, numero_ejemplos=5):
    """Muestra duplicados, coincidencias y claves sin correspondencia."""
    codigos_municipios = (
        municipios["cod_ine"].astype("string").str.strip().str.zfill(5)
    )
    vehiculos_municipales = vehiculos.loc[
        vehiculos["sin_especificar"].eq(False)
    ]
    codigos_vehiculos = (
        vehiculos_municipales["cod_ine"]
        .astype("string")
        .str.strip()
        .str.zfill(5)
    )
    codigos_vehiculos_todos = (
        vehiculos["cod_ine"].astype("string").str.strip().str.zfill(5)
    )
    codigos_puntos = (
        puntos_recarga["NATCODE"].astype("string").str.strip().str[-5:]
    )

    conjunto_municipios = set(codigos_municipios.dropna())
    conjunto_vehiculos = set(codigos_vehiculos.dropna())
    conjunto_puntos = set(codigos_puntos.dropna())

    print("\n=== Diagnóstico de claves y uniones ===")
    print("Códigos duplicados:")
    print(f"  municipios.cod_ine: {codigos_municipios.duplicated().sum()}")
    print(
        "  vehiculos.cod_ine (archivo completo): "
        f"{codigos_vehiculos_todos.duplicated().sum()}"
    )
    print(
        "  vehiculos.cod_ine (municipales): "
        f"{codigos_vehiculos.duplicated().sum()}"
    )
    print(f"  puntos_recarga.NATCODE: {puntos_recarga['NATCODE'].duplicated().sum()}")
    print(f"  NATCODE[-5:] usado en el join: {codigos_puntos.duplicated().sum()}")
    print(
        "  Filas de vehículos marcadas como sin especificar: "
        f"{vehiculos['sin_especificar'].eq(True).sum()}"
    )

    municipios_sin_vehiculos = conjunto_municipios - conjunto_vehiculos
    municipios_sin_puntos = conjunto_municipios - conjunto_puntos
    vehiculos_sin_municipio = conjunto_vehiculos - conjunto_municipios
    puntos_sin_municipio = conjunto_puntos - conjunto_municipios

    print("\nCoincidencias por código municipal único:")
    print(
        f"  Municipios con vehículos: "
        f"{len(conjunto_municipios & conjunto_vehiculos)} / "
        f"{len(conjunto_municipios)}; sin correspondencia: "
        f"{len(municipios_sin_vehiculos)}"
    )
    print(
        f"  Municipios con puntos de recarga: "
        f"{len(conjunto_municipios & conjunto_puntos)} / "
        f"{len(conjunto_municipios)}; sin correspondencia: "
        f"{len(municipios_sin_puntos)}"
    )
    print(
        "  Municipios presentes en las tres fuentes: "
        f"{len(conjunto_municipios & conjunto_vehiculos & conjunto_puntos)}"
    )
    print(f"  Códigos municipales de vehículos sin municipio: {len(vehiculos_sin_municipio)}")
    print(f"  Códigos candidatos de recarga sin municipio: {len(puntos_sin_municipio)}")

    ejemplos = [
        (
            "Municipios sin correspondencia en vehículos",
            municipios.loc[
                codigos_municipios.isin(municipios_sin_vehiculos),
                ["cod_ine", "nombre", "provincia"],
            ].head(numero_ejemplos),
        ),
        (
            "Municipios sin correspondencia en puntos de recarga",
            municipios.loc[
                codigos_municipios.isin(municipios_sin_puntos),
                ["cod_ine", "nombre", "provincia"],
            ].head(numero_ejemplos),
        ),
        (
            "Registros de recarga sin municipio correspondiente",
            puntos_recarga.loc[
                ~codigos_puntos.isin(conjunto_municipios),
                ["NATCODE", "NAMEUNIT"],
            ]
            .assign(
                codigo_candidato=codigos_puntos[
                    ~codigos_puntos.isin(conjunto_municipios)
                ].to_numpy()
            )
            .head(numero_ejemplos),
        ),
    ]

    for titulo, muestra in ejemplos:
        print(f"\n{titulo}:")
        print("  Ninguno" if muestra.empty else muestra.to_string(index=False))


def diagnosticar_nulos_indicadores(
    municipios, vehiculos, puntos_recarga, df_municipal, df_indicadores
):
    """Compara nulos originales, nulos tras el join y nulos de indicadores."""
    columnas_municipios = ["pob_2025", "pob_2020", "pernoctaciones_2025"]
    columnas_vehiculos = ["parque_total", "dist_0", "dist_eco"]
    columnas_puntos = ["refill_points_50kw"]
    vehiculos_municipales = vehiculos.loc[
        vehiculos["sin_especificar"].eq(False)
    ]

    print("\n=== Nulos en los Parquet originales ===")
    print("municipios:")
    print(municipios[columnas_municipios].isna().sum().to_string())
    print("\nvehiculos (archivo completo):")
    print(vehiculos[columnas_vehiculos].isna().sum().to_string())
    print("\nvehiculos (filas municipales usadas para el join):")
    print(vehiculos_municipales[columnas_vehiculos].isna().sum().to_string())
    print("\npuntos_recarga:")
    print(puntos_recarga[columnas_puntos].isna().sum().to_string())

    columnas_unidas = [
        "poblacion",
        "poblacion_5y",
        "pernoctaciones",
        "vehiculos",
        "vehiculos_cero_eco",
        "puntos_rapidos",
    ]
    print("\n=== Nulos en columnas después del join ===")
    print(df_municipal[columnas_unidas].isna().sum().to_string())

    print("\n=== Nulos en indicadores calculados ===")
    print(df_indicadores[INDICADORES].isna().sum().to_string())

    print("\nDenominadores en cero que pueden producir NaN:")
    for columna in ["poblacion", "poblacion_5y", "vehiculos"]:
        print(f"  {columna}: {df_municipal[columna].eq(0).sum()}")


def construir_dataframe_municipal(municipios, vehiculos, puntos_recarga):
    """Une las fuentes disponibles y normaliza sus columnas municipales."""
    fuentes = [
        ("municipios", municipios, [
            "cod_ine", "nombre", "provincia", "mas_10000",
            "pob_2025", "pob_2020",
            "pernoctaciones_2025",
        ]),
        ("vehiculos", vehiculos, [
            "cod_ine", "parque_total", "dist_0", "dist_eco",
            "sin_especificar",
        ]),
        ("puntos_recarga", puntos_recarga, [
            "NATCODE", "refill_points_50kw",
        ]),
    ]

    for nombre, df, columnas in fuentes:
        faltantes = sorted(set(columnas) - set(df.columns))
        if faltantes:
            raise ValueError(f"Faltan columnas en {nombre}: {faltantes}")

    df_municipios = municipios.rename(columns={
        "cod_ine": "codigo_ine",
        "nombre": "municipio",
        "pob_2025": "poblacion",
        "pob_2020": "poblacion_5y",
        "pernoctaciones_2025": "pernoctaciones",
    }).copy()
    df_municipios["codigo_ine"] = (
        df_municipios["codigo_ine"].astype("string").str.strip().str.zfill(5)
    )
    df_municipios = df_municipios.loc[
        df_municipios["mas_10000"].eq(True)
    ].copy()

    df_vehiculos = vehiculos.loc[
        vehiculos["sin_especificar"].eq(False)
    ].copy()
    df_vehiculos["codigo_ine"] = (
        df_vehiculos["cod_ine"].astype("string").str.strip().str.zfill(5)
    )
    df_vehiculos["vehiculos_cero_eco"] = (
        df_vehiculos["dist_0"] + df_vehiculos["dist_eco"]
    )

    df_puntos = puntos_recarga[["NATCODE", "refill_points_50kw"]].copy()
    df_puntos["codigo_ine"] = (
        df_puntos["NATCODE"].astype("string").str.strip().str[-5:]
    )
    df_puntos = df_puntos.rename(columns={
        "refill_points_50kw": "puntos_rapidos",
    })

    for nombre, df in [
        ("municipios", df_municipios),
        ("vehiculos", df_vehiculos),
        ("puntos_recarga", df_puntos),
    ]:
        if df["codigo_ine"].duplicated().any():
            raise ValueError(f"Hay códigos municipales duplicados en {nombre}.")

    df_municipal = (
        df_municipios
        .merge(
            df_vehiculos[[
                "codigo_ine", "parque_total", "vehiculos_cero_eco",
            ]].rename(columns={"parque_total": "vehiculos"}),
            on="codigo_ine",
            how="left",
            validate="one_to_one",
        )
        .merge(
            df_puntos[["codigo_ine", "puntos_rapidos"]],
            on="codigo_ine",
            how="left",
            validate="one_to_one",
        )
    )
    df_municipal["pernoctaciones"] = df_municipal[
        "pernoctaciones"
    ].fillna(0)

    return df_municipal


def revisar_datos(df):
    """Muestra información básica sobre el conjunto de datos."""
    print("\nDimensiones:", df.shape)

    print("\nPrimeras filas:")
    print(df.head())

    print("\nValores ausentes:")
    print(df.isna().sum())

    if "codigo_ine" in df:
        print("\nMunicipios duplicados:")
        print(df["codigo_ine"].duplicated().sum())


def calcular_indicadores(df):
    """Comprueba los datos y calcula los cinco indicadores municipales."""
    columnas = [
        "codigo_ine", "municipio", "provincia"
    ] + COLUMNAS_NUMERICAS

    faltantes = sorted(set(columnas) - set(df.columns))

    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    df = df.copy()

    # Mantener el código INE con cinco dígitos
    df["codigo_ine"] = (
        df["codigo_ine"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    if df["codigo_ine"].isna().any() or df["codigo_ine"].duplicated().any():
        raise ValueError("Hay códigos INE ausentes o duplicados.")

    if df[["municipio", "provincia"]].isna().any().any():
        raise ValueError("Hay municipios o provincias sin identificar.")

    # Convertir las columnas a valores numéricos
    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].apply(
        pd.to_numeric, errors="coerce"
    )

    if (df[COLUMNAS_NUMERICAS] < 0).any().any():
        raise ValueError("Hay valores negativos en los datos.")

    # Evitar divisiones entre cero
    poblacion = df["poblacion"].replace(0, np.nan)
    poblacion_5y = df["poblacion_5y"].replace(0, np.nan)
    vehiculos = df["vehiculos"].replace(0, np.nan)

    df["vehiculos_por_habitante"] = (
        df["vehiculos"] / poblacion
    )

    df["cero_eco_por_1000"] = (
        df["vehiculos_cero_eco"] / poblacion * 1000
    )

    df["puntos_rapidos_por_10000"] = (
        df["puntos_rapidos"] / vehiculos * 10000
    )

    df["crecimiento_poblacion"] = (
        (df["poblacion"] - df["poblacion_5y"])
        / poblacion_5y * 100
    )

    df["turismo_por_habitante"] = (
        df["pernoctaciones"] / poblacion
    )

    df[INDICADORES] = df[INDICADORES].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


def analizar_indicadores(df):
    """Analiza las distribuciones, valores ausentes y correlaciones."""
    print("\nEstadísticas descriptivas:")
    print(df[INDICADORES].describe().T.round(2).to_string())

    print("\nValores ausentes:")
    print(df[INDICADORES].isna().sum().to_string())

    correlaciones = df[INDICADORES].corr()

    print("\nCorrelaciones:")
    print(correlaciones.round(2).to_string())

    atipicos = detectar_valores_atipicos(df)
    print("\nValores atípicos por indicador (rango intercuartílico):")
    for indicador, municipios_atipicos in atipicos.items():
        porcentaje = len(municipios_atipicos) / len(df) * 100
        print(
            f"{indicador}: {len(municipios_atipicos)} "
            f"({porcentaje:.1f}%)"
        )
        if not municipios_atipicos.empty:
            print(
                municipios_atipicos
                .sort_values(indicador, ascending=False)
                .head(3)
                .to_string(index=False)
            )

    nombres_indicadores = {
        "vehiculos_por_habitante": "Vehículos por habitante",
        "cero_eco_por_1000": "Vehículos CERO/ECO por 1.000 habitantes",
        "puntos_rapidos_por_10000": "Puntos rápidos por 10.000 vehículos",
        "crecimiento_poblacion": "Crecimiento de población (%)",
        "turismo_por_habitante": "Pernoctaciones por habitante",
    }

    figura = plt.figure(figsize=(15, 8))
    figura.subplots_adjust(top=0.84)
    ax_histograma = figura.add_axes([0.07, 0.23, 0.40, 0.59])
    ax_boxplot = figura.add_axes([0.54, 0.23, 0.40, 0.59])
    ax_correlaciones = figura.add_axes([0.18, 0.22, 0.62, 0.61])
    ax_colorbar = figura.add_axes([0.83, 0.25, 0.03, 0.56])

    imagen = ax_correlaciones.imshow(
        correlaciones,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    ax_correlaciones.set_xticks(
        range(len(INDICADORES)),
        INDICADORES,
        rotation=45,
        ha="right"
    )

    ax_correlaciones.set_yticks(
        range(len(INDICADORES)),
        INDICADORES
    )

    for i in range(len(INDICADORES)):
        for j in range(len(INDICADORES)):
            valor = correlaciones.iloc[i, j]

            if pd.notna(valor):
                ax_correlaciones.text(
                    j, i, f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9
                )

    figura.colorbar(imagen, cax=ax_colorbar, label="Correlación")
    ax_correlaciones.set_visible(False)
    ax_colorbar.set_visible(False)

    boton_anterior_ax = figura.add_axes([0.40, 0.07, 0.08, 0.07])
    boton_siguiente_ax = figura.add_axes([0.52, 0.07, 0.08, 0.07])
    boton_anterior = Button(boton_anterior_ax, "←")
    boton_siguiente = Button(boton_siguiente_ax, "→")
    estado = {"pagina": 0}
    total_paginas = len(INDICADORES) + 1
    indicador_actual = figura.text(
        0.5, 0.16, "", ha="center", va="center"
    )

    def mostrar_pagina(pagina):
        estado["pagina"] = pagina
        ax_histograma.clear()
        ax_boxplot.clear()

        if pagina < len(INDICADORES):
            indicador = INDICADORES[pagina]
            nombre = nombres_indicadores[indicador]
            valores = df[indicador].dropna()

            ax_histograma.set_visible(True)
            ax_boxplot.set_visible(True)
            ax_correlaciones.set_visible(False)
            ax_colorbar.set_visible(False)

            ax_histograma.hist(valores, bins=30)
            ax_histograma.set_title("Histograma", fontsize=12)
            ax_histograma.set_xlabel(nombre)
            ax_histograma.set_ylabel("Número de municipios")
            ax_histograma.grid(axis="y", alpha=0.25)

            if not valores.empty:
                ax_boxplot.boxplot(valores, vert=False)

            ax_boxplot.set_title("Boxplot", fontsize=12)
            ax_boxplot.set_xlabel(nombre)
            ax_boxplot.set_yticks([])
            ax_boxplot.grid(axis="x", alpha=0.25)
            indicador_actual.set_text(
                f"{nombre} ({pagina + 1}/{total_paginas})"
            )
        else:
            ax_histograma.set_visible(False)
            ax_boxplot.set_visible(False)
            ax_correlaciones.set_visible(True)
            ax_colorbar.set_visible(True)
            indicador_actual.set_text(
                f"Matriz de correlaciones ({pagina + 1}/{total_paginas})"
            )

        figura.canvas.draw_idle()

    def cambiar_pagina(direccion):
        mostrar_pagina(
            (estado["pagina"] + direccion) % total_paginas
        )

    def navegar_con_teclado(evento):
        if evento.key == "left":
            cambiar_pagina(-1)
        elif evento.key == "right":
            cambiar_pagina(1)

    boton_anterior.on_clicked(lambda _: cambiar_pagina(-1))
    boton_siguiente.on_clicked(lambda _: cambiar_pagina(1))
    callback_teclado = figura.canvas.mpl_connect(
        "key_press_event", navegar_con_teclado
    )
    figura._navegador_indicadores = (
        boton_anterior,
        boton_siguiente,
        callback_teclado,
        estado,
    )
    mostrar_pagina(0)

    return [figura]


def detectar_valores_atipicos(df):
    """Detecta valores atípicos mediante el rango intercuartílico."""
    resultados = {}

    for indicador in INDICADORES:
        q1 = df[indicador].quantile(0.25)
        q3 = df[indicador].quantile(0.75)
        iqr = q3 - q1

        minimo = q1 - 1.5 * iqr
        maximo = q3 + 1.5 * iqr

        resultados[indicador] = df.loc[
            (df[indicador] < minimo) | (df[indicador] > maximo),
            ["municipio", indicador],
        ].copy()

    return resultados


def normalizar_indicadores(df):
    """Normaliza los indicadores entre 0 y 1."""
    if df[INDICADORES].isna().any().any():
        raise ValueError(
            "Hay indicadores con valores ausentes. Revisa los datos."
        )

    df = df.copy()

    scaler = MinMaxScaler()
    df[INDICADORES] = scaler.fit_transform(df[INDICADORES])

    # Menos cargadores existentes representa mayor oportunidad
    df["puntos_rapidos_por_10000"] = (
        1 - df["puntos_rapidos_por_10000"]
    )

    return df


def calcular_ranking(df, pesos=PESOS):
    """Calcula el Opportunity Score y ordena los municipios."""
    if set(pesos) != set(INDICADORES):
        raise ValueError(
            "Los pesos deben incluir los cinco indicadores."
        )

    if any(p < 0 for p in pesos.values()) or not np.isclose(
        sum(pesos.values()), 1
    ):
        raise ValueError(
            "Los pesos deben ser no negativos y sumar 1."
        )

    if "mas_10000" not in df.columns:
        raise ValueError(
            "El ranking requiere la columna 'mas_10000'."
        )

    df = df.loc[df["mas_10000"].eq(True)].copy()

    df["opportunity_score"] = sum(
        df[indicador] * peso
        for indicador, peso in pesos.items()
    ) * 100

    return df.sort_values(
        ["opportunity_score", "codigo_ine"],
        ascending=[False, True],
    ).reset_index(drop=True)


def mostrar_ranking(ranking, numero=20):
    """Crea el gráfico de los municipios con mayor Opportunity Score."""
    mejores = ranking.head(numero).sort_values("opportunity_score")
    posiciones = np.arange(len(mejores))
    municipios = mejores["municipio"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        posiciones,
        mejores["opportunity_score"]
    )

    ax.set_yticks(posiciones, labels=municipios)
    ax.set_xlabel("Opportunity Score")
    ax.set_ylabel("Municipio")
    ax.set_title(f"Top {numero} municipios por Opportunity Score")
    ax.set_xlim(0, 100)

    fig.tight_layout()

    return fig


def seleccionar_top20(ranking):
    """Selecciona hasta 20 municipios, con máximo 3 por provincia."""
    candidatos = ranking.dropna(
        subset=["codigo_ine", "provincia", "opportunity_score"]
    ).copy()

    candidatos = candidatos.sort_values(
        ["opportunity_score", "codigo_ine"],
        ascending=[False, True],
    ).drop_duplicates("codigo_ine")

    candidatos["posicion_ranking"] = range(
        1, len(candidatos) + 1
    )

    seleccionados = []
    provincias = {}

    for _, municipio in candidatos.iterrows():
        provincia = municipio["provincia"]

        if provincias.get(provincia, 0) >= 3:
            continue

        seleccionados.append(municipio)

        provincias[provincia] = provincias.get(provincia, 0) + 1

        if len(seleccionados) == 20:
            break

    return pd.DataFrame(
        seleccionados,
        columns=candidatos.columns,
    ).reset_index(drop=True)


def comparar_con_poblacion(df, ranking):
    """Compara el ranking del modelo con el ranking por población."""
    poblacion = df[
        ["codigo_ine", "municipio", "poblacion"]
    ].drop_duplicates("codigo_ine").copy()

    poblacion = poblacion.sort_values(
        ["poblacion", "codigo_ine"],
        ascending=[False, True]
    )

    poblacion["posicion_poblacion"] = range(
        1, len(poblacion) + 1
    )

    modelo = ranking[
        ["codigo_ine", "opportunity_score"]
    ].copy()

    modelo["posicion_modelo"] = range(
        1, len(modelo) + 1
    )

    comparacion = poblacion.merge(
        modelo, on="codigo_ine"
    )

    comparacion["cambio_posicion"] = (
        comparacion["posicion_poblacion"]
        - comparacion["posicion_modelo"]
    )

    return comparacion.sort_values("posicion_modelo")


def analizar_sensibilidad(df, escenarios=ESCENARIOS):
    """Compara los rankings al modificar los pesos."""
    if "mas_10000" not in df.columns:
        raise ValueError(
            "El ranking requiere la columna 'mas_10000'."
        )

    df = df.loc[df["mas_10000"].eq(True)].copy()
    resultados = df[
        ["codigo_ine", "municipio", "provincia"]
    ].copy()

    for nombre, pesos in escenarios.items():
        ranking = calcular_ranking(df, pesos)

        ranking["posicion"] = range(
            1, len(ranking) + 1
        )

        top20 = seleccionar_top20(ranking)

        datos = ranking[
            ["codigo_ine", "opportunity_score", "posicion"]
        ].copy()

        datos["seleccionado"] = datos["codigo_ine"].isin(
            top20["codigo_ine"]
        )

        datos = datos.rename(columns={
            "opportunity_score": f"score_{nombre}",
            "posicion": f"posicion_{nombre}",
            "seleccionado": f"seleccionado_{nombre}",
        })

        resultados = resultados.merge(
            datos,
            on="codigo_ine",
            how="left"
        )

    return resultados


def evaluar_clusters(df):
    """Calcula la inercia y silueta para distintos números de clusters."""
    if len(df) < 3:
        raise ValueError(
            "Se necesitan al menos tres municipios."
        )

    max_k = min(10, len(df) - 1)
    resultados = []

    for k in range(1, max_k + 1):
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        etiquetas = modelo.fit_predict(df[INDICADORES])

        silueta = np.nan

        if 1 < len(set(etiquetas)) < len(df):
            silueta = silhouette_score(
                df[INDICADORES], etiquetas
            )

        resultados.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silueta,
        })

    return pd.DataFrame(resultados)


def mostrar_evaluacion_clusters(evaluacion):
    """Crea los gráficos del codo y del coeficiente de silueta."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    ejes[0].plot(
        evaluacion["k"], evaluacion["inercia"], marker="o"
    )
    ejes[0].set_xlabel("Número de clusters (k)")
    ejes[0].set_ylabel("Inercia")
    ejes[0].set_title("Método del codo")

    ejes[1].plot(
        evaluacion["k"], evaluacion["silueta"], marker="o"
    )
    ejes[1].set_xlabel("Número de clusters (k)")
    ejes[1].set_ylabel("Coeficiente de silueta")
    ejes[1].set_title("Coeficiente de silueta")

    fig.tight_layout()

    return [fig]


def agrupar_municipios(df, numero_grupos=3):
    """Agrupa municipios similares mediante K-means."""
    if not 2 <= numero_grupos < len(df):
        raise ValueError(
            "El número de grupos debe estar entre 2 y n-1."
        )

    df = df.copy()

    modelo = KMeans(
        n_clusters=numero_grupos,
        random_state=42,
        n_init=10
    )

    df["cluster"] = modelo.fit_predict(df[INDICADORES])

    return df


def mostrar_clusters(clusters):
    """Crea un gráfico de dispersión de los clusters."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for grupo, datos in clusters.groupby("cluster"):
        ax.scatter(
            datos["vehiculos_por_habitante"],
            datos["cero_eco_por_1000"],
            label=f"Cluster {grupo}",
            alpha=0.7
        )

    ax.set_xlabel("Vehículos por habitante")
    ax.set_ylabel("CERO/ECO por 1.000 habitantes")
    ax.set_title("Segmentación de municipios")
    ax.legend()

    fig.tight_layout()

    return fig
