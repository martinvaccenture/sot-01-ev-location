
"""
Pruebas del modelo EV Location Intelligence.

Genera la tabla Gold y ejecuta los análisis exploratorios.
Todas las funciones están implementadas en gold.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from gold import (
    generar_gold,
    comprobar_gold,
    analizar_atipicos,
    analizar_sensibilidad,
    analizar_posibles_flotas,
    comparar_con_baseline,
    graficos_indicadores,
    grafico_comparacion,
    analizar_kmeans,
    graficos_kmeans,
)


# Rutas del proyecto.
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ / "data"
RUTA_GOLD = CARPETA_DATOS / "gold.parquet"

PESOS_MODELO = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
ESCENARIOS_SENSIBILIDAD = {
    "Actual": PESOS_MODELO,
    "Equilibrado": [0.25, 0.25, 0.25, 0.25],
    "Electrificación": [0.15, 0.50, 0.20, 0.15],
    "Infraestructura": [0.15, 0.20, 0.50, 0.15],
    "Crecimiento": [0.15, 0.20, 0.20, 0.45],
}


def main(pesos=None, escenarios=None):
    pesos = PESOS_MODELO if pesos is None else pesos
    escenarios = (
        ESCENARIOS_SENSIBILIDAD if escenarios is None else escenarios
    )

    print("Pesos del modelo:")
    print([round(peso * 100, 1) for peso in pesos], "%")

    # 1. Generar y guardar la tabla Gold.
    gold = generar_gold(CARPETA_DATOS, RUTA_GOLD, pesos)

    print(f"\nTabla Gold generada: {RUTA_GOLD}")

    # 2. Comprobar calidad y analizar valores atípicos.
    comprobar_gold(gold)
    analizar_atipicos(gold)

    # 3. Revisar posibles flotas, sin excluir municipios.
    analizar_posibles_flotas(gold)

    # 4. Obtener el Top 20 y compararlo con el baseline.
    modelo, baseline = comparar_con_baseline(gold)

    # 5. Analizar la estabilidad del Top 20 ante cambios de pesos.
    analizar_sensibilidad(gold, escenarios)

    # 6. Primera ventana: histogramas, boxplots y correlación.
    graficos_indicadores(gold)

    # 7. Segunda ventana: baseline y ranking del modelo.
    grafico_comparacion(modelo, baseline)

    # 8. Tercera ventana: análisis de K-Means.
    resultado_kmeans, metricas = analizar_kmeans(gold)
    graficos_kmeans(resultado_kmeans, metricas)

    # 9. Mostrar las tres ventanas.
    plt.show()


if __name__ == "__main__":
    main()
