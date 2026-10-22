
"""
Pruebas del modelo EV Location Intelligence.
Genera Gold, ejecuta los análisis y prepara los rankings para Streamlit.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from gold import (
    generar_gold, comprobar_gold, analizar_atipicos,
    analizar_sensibilidad, analizar_posibles_flotas,
    comparar_con_baseline, graficos_indicadores,
    grafico_comparacion, analizar_kmeans, graficos_kmeans,
    preparar_escenarios, consolidar_escenarios,
)


RAIZ = Path(__file__).resolve().parents[2]
CARPETA_DATOS = RAIZ / "data"

RUTA_GOLD = CARPETA_DATOS / "gold.parquet"
RUTA_TURISMO = CARPETA_DATOS / "ranking_turismo.parquet"

NORMALIZACION = "p5_p95"

PESOS_MODELO = [1 / 6, 1 / 3, 1 / 3, 1 / 6]

ESCENARIOS_SENSIBILIDAD = {
    "Actual": PESOS_MODELO,
    "Equilibrado": [0.25, 0.25, 0.25, 0.25],
    "Electrificación": [0.15, 0.50, 0.20, 0.15],
    "Infraestructura": [0.15, 0.20, 0.50, 0.15],
    "Crecimiento": [0.15, 0.20, 0.20, 0.45],
}


def main(pesos=None, escenarios=None, normalizacion=NORMALIZACION):

    pesos = PESOS_MODELO if pesos is None else pesos

    escenarios = (
        ESCENARIOS_SENSIBILIDAD.copy()
        if escenarios is None else dict(escenarios)
    )

    escenarios["Actual"] = pesos

    print("Pesos del modelo:")
    print([round(p * 100, 1) for p in pesos], "%")
    print(f"Normalización: {normalizacion}")

    # 1. Generar Gold.
    gold = generar_gold(
        CARPETA_DATOS,
        pesos=pesos,
        normalizacion=normalizacion
    )

    # 2. Calidad y valores atípicos.
    comprobar_gold(gold)
    analizar_atipicos(gold)

    # 3. Posibles flotas.
    analizar_posibles_flotas(gold)

    # 4. Ranking y baseline.
    modelo, baseline = comparar_con_baseline(gold)

    # 5. Análisis de sensibilidad.
    analizar_sensibilidad(
        gold,
        escenarios,
        mostrar_cambios=True
    )

    # 6. Gráficos de indicadores y rankings.
    graficos_indicadores(gold)
    grafico_comparacion(modelo, baseline)

    # 7. K-Means.
    gold, metricas = analizar_kmeans(gold)
    graficos_kmeans(gold, metricas)

    # 8. Crear las dos tablas de cada escenario.
    tablas_escenarios = preparar_escenarios(
        gold,
        escenarios,
        normalizacion=normalizacion
    )

    for nombre, tablas in tablas_escenarios.items():

        print(f"\n{'=' * 75}")
        print(f"ESCENARIO: {nombre}")
        print(f"{'=' * 75}")

        print("\nTABLA 1: CUATRO INDICADORES")
        print(f"Municipios: {len(tablas['cuatro'])}")

        print(
            tablas["cuatro"][
                [
                    "municipio",
                    "provincia",
                    "opportunity_score",
                    "ranking_general",
                    "top20",
                    "apariciones_escenarios"
                ]
            ].head(30).to_string(index=False)
        )

        print("\nTABLA 2: TURISMO")
        print(f"Municipios: {len(tablas['turismo'])}")

        print(
            tablas["turismo"][
                [
                    "municipio",
                    "provincia",
                    "pernoctaciones_2025",
                    "posicion_inicial",
                    "score_final",
                    "posicion_final",
                    "top20_final",
                    "cambio_posicion",
                    "apariciones_finales"
                ]
            ].to_string(index=False)
        )

    # 9. Consolidar las tablas para Streamlit.
    gold_streamlit, turismo_streamlit = consolidar_escenarios(
        gold,
        tablas_escenarios
    )

    # 10. Guardar únicamente dos archivos Parquet.
    RUTA_GOLD.parent.mkdir(parents=True, exist_ok=True)

    gold_streamlit.to_parquet(RUTA_GOLD, index=False)
    turismo_streamlit.to_parquet(RUTA_TURISMO, index=False)

    print(f"\nGold guardado en: {RUTA_GOLD}")
    print(f"Ranking de turismo guardado en: {RUTA_TURISMO}")

    # 11. Mostrar gráficos.
    plt.show()

    return tablas_escenarios


if __name__ == "__main__":
    tablas_escenarios = main()
