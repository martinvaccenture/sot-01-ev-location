"""Pruebas del modelo EV Location Intelligence y preparación de rankings."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gold import (
    analizar_atipicos,
    analizar_kmeans,
    analizar_posibles_flotas,
    analizar_sensibilidad,
    comparar_con_baseline,
    comprobar_gold,
    generar_gold,
    grafico_comparacion,
    graficos_indicadores,
    graficos_kmeans,
    preparar_escenarios,
)


RAIZ = Path(__file__).resolve().parents[2]
CARPETA_DATOS = RAIZ / "data"
RUTA_GOLD = CARPETA_DATOS / "gold.parquet"
RUTA_TURISMO = CARPETA_DATOS / "ranking_turismo.parquet"
CARPETA_RESULTADOS = RAIZ / "resultados"

NORMALIZACION = "p5_p95"
PESOS_MODELO = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
ESCENARIOS_SENSIBILIDAD = {
    "Actual": PESOS_MODELO,
    "Equilibrado": [0.25, 0.25, 0.25, 0.25],
    "Electrificación": [0.15, 0.50, 0.20, 0.15],
    "Infraestructura": [0.15, 0.20, 0.50, 0.15],
    "Crecimiento": [0.15, 0.20, 0.20, 0.45],
}

# Valor prudente porque actualmente solo una parte de los municipios tiene
# pernoctaciones observadas. Se puede sustituir al llamar a main().
PESO_TURISMO = 0.10

# Umbrales de alerta revisables; no confirman que exista una flota empresarial.
UMBRALES_RIESGO_FLOTA = {
    "vehiculos_por_habitante": 2.0,
    "cero_eco_por_1000": 200.0,
}

RUTAS_TABLAS_FINALES = {
    "sin_turismo": CARPETA_RESULTADOS / "top20_sin_turismo.parquet",
    "con_turismo": CARPETA_RESULTADOS / "top20_con_turismo.parquet",
    "sin_turismo_sin_flotas": (
        CARPETA_RESULTADOS / "top20_sin_turismo_sin_flotas.parquet"
    ),
    "con_turismo_sin_flotas": (
        CARPETA_RESULTADOS / "top20_con_turismo_sin_flotas.parquet"
    ),
}


def main(
    pesos=None,
    escenarios=None,
    peso_turismo=PESO_TURISMO,
    umbrales_flotas=None,
    normalizacion=NORMALIZACION,
):
    """Ejecuta el análisis y deja configurables pesos, escenarios y filtros."""
    pesos = PESOS_MODELO if pesos is None else pesos
    escenarios = (
        ESCENARIOS_SENSIBILIDAD.copy()
        if escenarios is None else dict(escenarios)
    )
    escenarios["Actual"] = pesos
    umbrales_flotas = (
        UMBRALES_RIESGO_FLOTA.copy()
        if umbrales_flotas is None else dict(umbrales_flotas)
    )

    print("Pesos del modelo actual:")
    print([round(peso * 100, 1) for peso in pesos], "%")
    print(f"Normalización: {normalizacion}")
    print(f"Peso de turismo con dato válido: {peso_turismo:.1%}")
    print(f"Escenarios base: {len(escenarios)}; con turismo: {len(escenarios)}")

    gold = generar_gold(
        CARPETA_DATOS,
        pesos=pesos,
        normalizacion=normalizacion,
    )
    comprobar_gold(gold)
    analizar_atipicos(gold)
    analizar_posibles_flotas(gold, umbrales_flotas)

    modelo, baseline = comparar_con_baseline(gold)
    graficos_indicadores(gold)
    grafico_comparacion(modelo, baseline)

    gold, metricas = analizar_kmeans(gold)
    graficos_kmeans(gold, metricas)

    preparacion = preparar_escenarios(
        gold,
        escenarios,
        peso_turismo,
        normalizacion,
        umbrales_flotas,
        escenario_referencia="Actual",
    )
    analizar_sensibilidad(preparacion)

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    preparacion["gold"].to_parquet(RUTA_GOLD, index=False)
    columnas_tabla = [
        "posicion_top20", "municipio", "provincia", "poblacion",
        "opportunity_score", "ranking_general", "aparicion_escenarios",
        "posible_riesgo_flota",
    ]
    for nombre, tabla_final in preparacion["tablas_finales"].items():
        print(f"\n=== TABLA FINAL: {nombre} ===")
        print(f"Municipios: {len(tabla_final)}")
        print(tabla_final[columnas_tabla].to_string(index=False))
        tabla_final.to_parquet(RUTAS_TABLAS_FINALES[nombre], index=False)

    ranking_turismo = pd.concat(
        [
            preparacion["tablas_finales"]["con_turismo"].assign(
                tabla_final="con_turismo"
            ),
            preparacion["tablas_finales"]["con_turismo_sin_flotas"].assign(
                tabla_final="con_turismo_sin_flotas"
            ),
        ],
        ignore_index=True,
    )
    ranking_turismo.to_parquet(RUTA_TURISMO, index=False)

    print(f"\nGold global guardado en: {RUTA_GOLD}")
    print(f"Tablas finales guardadas en: {CARPETA_RESULTADOS}")
    print(f"Ranking de turismo guardado en: {RUTA_TURISMO}")

    plt.show()
    return preparacion


if __name__ == "__main__":
    resultado = main()
