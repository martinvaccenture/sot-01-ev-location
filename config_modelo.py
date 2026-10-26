# ============================================================
# CONFIGURACIÓN GENERAL DEL MODELO
# ============================================================

NORMALIZACION = "p5_p95"


# ============================================================
# PESOS DEL MODELO BASE
# ============================================================

PESOS_MODELO = [
    1 / 6,
    1 / 3,
    1 / 3,
    1 / 6
]


# ============================================================
# ESCENARIOS
# ============================================================

ESCENARIOS = {

    "modelo_base": {
        "pesos": PESOS_MODELO,
        "peso_turismo": 0.10
    },

    "electrificacion": {
        "pesos": [
            0.15,
            0.50,
            0.20,
            0.15
        ],
        "peso_turismo": 0.08
    },

    "infraestructura": {
        "pesos": [
            0.15,
            0.20,
            0.50,
            0.15
        ],
        "peso_turismo": 0.12
    },

    "crecimiento": {
        "pesos": [
            0.15,
            0.20,
            0.20,
            0.45
        ],
        "peso_turismo": 0.06
    }
}


# ============================================================
# CONFIGURACIÓN RIESGO DE FLOTA
# ============================================================

VALORES_RIESGO_EXCLUIDOS = (
    "riesgo alto",
    "posible riesgo"
)


UMBRALES_RIESGO_FLOTA = {
    "vehiculos_por_habitante": 2.0,
    "cero_eco_por_1000": 200.0
}