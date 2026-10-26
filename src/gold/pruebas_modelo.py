"""Pruebas del modelo EV Location Intelligence y preparación de rankings."""

from pathlib import Path

import matplotlib.pyplot as plt

from config_modelo import (
    NORMALIZACION,
    PESOS_MODELO,
    ESCENARIOS,
    VALORES_RIESGO_EXCLUIDOS,
    UMBRALES_RIESGO_FLOTA,
)

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
CARPETA_RESULTADOS = RAIZ / "resultados"


def main(
    pesos=None,
    escenarios=None,
    umbrales_flotas=None,
    valores_riesgo_excluidos=VALORES_RIESGO_EXCLUIDOS,
    normalizacion=NORMALIZACION,
):
    """Ejecuta el análisis y deja configurables pesos, escenarios y filtros."""

    pesos = PESOS_MODELO if pesos is None else pesos

    escenarios = {
        nombre: dict(config)
        for nombre, config in (
            ESCENARIOS if escenarios is None else escenarios
        ).items()
    }

    umbrales_flotas = (
        UMBRALES_RIESGO_FLOTA.copy()
        if umbrales_flotas is None
        else dict(umbrales_flotas)
    )


    print("Pesos del modelo base:")
    print(
        [round(peso * 100, 1) for peso in pesos],
        "%"
    )

    print(f"Normalización: {normalizacion}")

    print(
        f"Escenarios: {len(escenarios)} sin turismo "
        f"y {len(escenarios)} con turismo"
    )


    gold = generar_gold(
        CARPETA_DATOS,
        pesos=pesos,
        normalizacion=normalizacion,
    )


    comprobar_gold(gold)

    analizar_atipicos(gold)

    analizar_posibles_flotas(
        gold,
        umbrales_flotas
    )


    modelo, baseline = comparar_con_baseline(
        gold
    )

    graficos_indicadores(
        gold
    )

    grafico_comparacion(
        modelo,
        baseline
    )


    resultados_kmeans, metricas = analizar_kmeans(
        gold
    )

    graficos_kmeans(
        resultados_kmeans,
        metricas
    )


    preparacion = preparar_escenarios(
        gold,
        escenarios,
        normalizacion,
        umbrales_flotas,
        valores_riesgo_excluidos,
    )


    analizar_sensibilidad(
        preparacion
    )


    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )


    gold.to_parquet(
        RUTA_GOLD,
        index=False
    )


    tablas = {
        **preparacion["tablas_escenarios"],
        **preparacion["tablas_top"]
    }


    for nombre, tabla_final in tablas.items():

        tabla_final.to_parquet(
            CARPETA_RESULTADOS / f"{nombre}.parquet",
            index=False
        )


    for nombre, top in preparacion["tablas_top"].items():

        print(
            f"\n=== {nombre.upper()} ==="
        )

        print(
            top.to_string(
                index=False
            )
        )


    print(
        f"\nGold global guardado en: "
        f"{RUTA_GOLD}"
    )

    print(
        f"{len(tablas)} tablas guardadas en: "
        f"{CARPETA_RESULTADOS}"
    )


    plt.show()

    return preparacion


if __name__ == "__main__":
    resultado = main()