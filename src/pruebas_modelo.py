from pathlib import Path

import matplotlib.pyplot as plt

from gold import (
    ESCENARIOS,
    INDICADORES,
    PESOS,
    analizar_sensibilidad,
    analizar_indicadores,
    calcular_indicadores,
    calcular_ranking,
    cargar_parquet,
    comparar_con_poblacion,
    construir_dataframe_municipal,
    diagnosticar_nulos_indicadores,
    diagnosticar_uniones,
    mostrar_resumen_dataframe,
    normalizar_indicadores,
    seleccionar_top20,
)


CARPETA_DATOS = Path(__file__).resolve().parents[1] / "data"
ARCHIVOS_PARQUET = {
    "municipios": CARPETA_DATOS / "municipios.parquet",
    "vehiculos": CARPETA_DATOS / "vehiculos.parquet",
    "puntos_recarga": CARPETA_DATOS / "puntos_recarga.parquet",
}


def main():
    dataframes = {}

        # Carga cada fuente y muestra su estructura y valores ausentes.
    for nombre, ruta in ARCHIVOS_PARQUET.items():
        df = cargar_parquet(ruta)
        mostrar_resumen_dataframe(df, nombre)
        dataframes[nombre] = df

        # Revisa las claves antes de unir las tres fuentes.
    diagnosticar_uniones(
        dataframes["municipios"],
        dataframes["vehiculos"],
        dataframes["puntos_recarga"],
    )

        # Construye los datos municipales y calcula los cinco indicadores.
    df_municipal = construir_dataframe_municipal(
        dataframes["municipios"],
        dataframes["vehiculos"],
        dataframes["puntos_recarga"],
    )
    df_indicadores = calcular_indicadores(df_municipal)
    diagnosticar_nulos_indicadores(
        dataframes["municipios"],
        dataframes["vehiculos"],
        dataframes["puntos_recarga"],
        df_municipal,
        df_indicadores,
    )
    # Muestra estadísticas, correlaciones, atípicos y gráficas navegables.
    analizar_indicadores(df_indicadores)

    # Normaliza los indicadores y calcula el ranking con los pesos base.
    df_normalizado = normalizar_indicadores(df_indicadores)
    ranking_base = calcular_ranking(df_normalizado, PESOS)
    top20_base = seleccionar_top20(ranking_base)
        # Usa el ranking por población como referencia para el score base.
    comparacion_baseline = comparar_con_poblacion(
        df_indicadores, ranking_base
    )
        # Recalcula los rankings con los mismos datos y distintos pesos.
    resultados_sensibilidad = analizar_sensibilidad(
        df_normalizado, ESCENARIOS
    )

    print("\n=== Pesos de los escenarios ===")
    for nombre_escenario, pesos in ESCENARIOS.items():
        print(f"\n{nombre_escenario}:")
        for indicador, peso in pesos.items():
            print(f"  {indicador}: {peso:.0%}")
        print(f"  Suma: {sum(pesos.values()):.0%}")

    print("\n=== Opportunity Score base ===")
    print(f"Municipios evaluados: {len(ranking_base):,}")
    print(
        "Rango de scores: "
        f"{ranking_base['opportunity_score'].min():.2f} - "
        f"{ranking_base['opportunity_score'].max():.2f}"
    )
    print("Top 20, con máximo 3 municipios por provincia:")
    print(
        top20_base[[
            "codigo_ine", "municipio", "provincia",
            "opportunity_score", "posicion_ranking",
        ]].to_string(index=False)
    )

    baseline_poblacion = comparacion_baseline.nsmallest(
        20, "posicion_poblacion"
    )
    codigos_baseline = set(baseline_poblacion["codigo_ine"])
    coincidencias_score = set(
        ranking_base.head(20)["codigo_ine"]
    ) & codigos_baseline
    coincidencias_seleccion = set(
        top20_base["codigo_ine"]
    ) & codigos_baseline
    print("\n=== Baseline: ranking por población ===")
    print(
        "Coincidencias entre Top 20 bruto por score y población: "
        f"{len(coincidencias_score)} / 20"
    )
    print(
        "Coincidencias con el Top 20 score limitado a 3 por provincia: "
        f"{len(coincidencias_seleccion)} / 20"
    )
    print(
        baseline_poblacion[[
            "codigo_ine", "municipio", "poblacion",
            "posicion_poblacion", "opportunity_score",
            "posicion_modelo", "cambio_posicion",
        ]].to_string(index=False)
    )

    seleccionados_base = set(
        resultados_sensibilidad.loc[
            resultados_sensibilidad["seleccionado_Base"], "codigo_ine"
        ]
    )
    print("\n=== Sensibilidad del Top 20 ===")
    for nombre_escenario in ESCENARIOS:
        seleccionados = set(
            resultados_sensibilidad.loc[
                resultados_sensibilidad[
                    f"seleccionado_{nombre_escenario}"
                ],
                "codigo_ine",
            ]
        )
        print(
            f"{nombre_escenario}: {len(seleccionados)} seleccionados; "
            f"{len(seleccionados & seleccionados_base)} coinciden con Base"
        )

    print("\n=== DataFrame municipal con indicadores ===")
    print(f"Filas: {len(df_indicadores):,}")
    print(
        df_indicadores[
            ["codigo_ine", "municipio", "provincia", *INDICADORES]
        ].head().to_string(index=False)
    )

    plt.show()
    return df_indicadores


if __name__ == "__main__":
    main()