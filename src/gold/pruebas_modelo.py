"""Pruebas del modelo EV Location Intelligence y preparación de rankings."""

from pathlib import Path

import matplotlib.pyplot as plt

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

NORMALIZACION = "p5_p95"
PESOS_MODELO = [1 / 6, 1 / 3, 1 / 3, 1 / 6]

# Orden de pesos: vehículos/hab., CERO/ECO, escasez de recarga rápida, crecimiento.
# peso_turismo solo se aplica a municipios con pernoctaciones (118 de 783),
# por eso se mantiene pequeño y se ajusta a la lógica de cada escenario.
ESCENARIOS = {
    # Referencia: los visitantes añaden demanda de recarga, pero no es el foco.
    "modelo_base": {"pesos": PESOS_MODELO, "peso_turismo": 0.10},
    # El parque CERO/ECO es residente; el visitante no aparece en ese indicador.
    "electrificacion": {"pesos": [0.15, 0.50, 0.20, 0.15], "peso_turismo": 0.08},
    # El visitante carga en ruta y sin cargador doméstico: depende de la red rápida.
    "infraestructura": {"pesos": [0.15, 0.20, 0.50, 0.15], "peso_turismo": 0.12},
    # Se busca demanda residente estable; la turística es estacional.
    "crecimiento": {"pesos": [0.15, 0.20, 0.20, 0.45], "peso_turismo": 0.06},
}

# Valores de riesgo_flota (vehiculos.parquet) que se eliminan en las tablas de confianza.
VALORES_RIESGO_EXCLUIDOS = ("riesgo alto", "posible riesgo")

# Solo se usan mientras vehiculos.parquet no tenga la columna riesgo_flota.
UMBRALES_RIESGO_FLOTA = {
    "vehiculos_por_habitante": 2.0,
    "cero_eco_por_1000": 200.0,
}


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
        for nombre, config in (ESCENARIOS if escenarios is None else escenarios).items()
    }
    umbrales_flotas = (
        UMBRALES_RIESGO_FLOTA.copy()
        if umbrales_flotas is None else dict(umbrales_flotas)
    )

    print("Pesos del modelo base:")
    print([round(peso * 100, 1) for peso in pesos], "%")
    print(f"Normalización: {normalizacion}")
    print(f"Escenarios: {len(escenarios)} sin turismo y {len(escenarios)} con turismo")

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
        normalizacion,
        umbrales_flotas,
        valores_riesgo_excluidos,
    )
    analizar_sensibilidad(preparacion)

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    preparacion["gold"].to_parquet(RUTA_GOLD, index=False)

    tablas = {**preparacion["tablas_escenarios"], **preparacion["tablas_top"]}
    for nombre, tabla_final in tablas.items():
        tabla_final.to_parquet(CARPETA_RESULTADOS / f"{nombre}.parquet", index=False)

    for nombre, top in preparacion["tablas_top"].items():
        print(f"\n=== {nombre.upper()} ===")
        print(top.to_string(index=False))

    print(f"\nGold global guardado en: {RUTA_GOLD}")
    print(f"{len(tablas)} tablas guardadas en: {CARPETA_RESULTADOS}")

    plt.show()
    return preparacion


if __name__ == "__main__":
    resultado = main()
