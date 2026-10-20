# EV Location Intelligence

## 1. Problema y decisión

**EV Location Intelligence** busca apoyar la decisión de **en qué municipios españoles priorizar la instalación de nuevos hubs de recarga rápida para vehículos eléctricos**.

La decisión se apoya en datos demográficos, parque de vehículos, infraestructura de recarga existente e información geográfica, con el objetivo de identificar y ordenar municipios según su potencial.

El resultado del proyecto será una priorización de municipios que permita apoyar una decisión de localización basada en datos.

---

## 2. Hipótesis

Antes de realizar el análisis se plantean las siguientes hipótesis:

- Los municipios con mayor población presentan una mayor demanda potencial.
- Una mayor presencia de vehículos, especialmente eléctricos, incrementa la necesidad potencial de infraestructura de recarga.
- La infraestructura de recarga ya existente debe considerarse para identificar tanto cobertura como posibles carencias.
- La disponibilidad de puntos de recarga rápida, especialmente de potencia ≥ 50 kW, puede ayudar a caracterizar el nivel actual de infraestructura.
- La combinación de diferentes variables puede ofrecer una priorización más útil que analizar cada indicador de forma independiente.
- Municipios con características similares pueden agruparse para identificar diferentes tipologías de oportunidad.

Estas hipótesis serán contrastadas mediante el scoring y el análisis de clustering.

---

## 3. Fuentes

| Fuente | Información utilizada | Fecha descarga | Licencia |
|---|---|---|---|
| INE | Información demográfica y municipal | TODO | TODO |
| DGT | Parque de vehículos e información relacionada con vehículos eléctricos | TODO | TODO |
| DGT NAP | Infraestructura y puntos de recarga | TODO | TODO |
| CNIG | Información geográfica y territorial | TODO | TODO |

La integración de las fuentes se realizará utilizando identificadores territoriales comunes, principalmente el **código INE municipal**.

---

## 4. Calidad de datos

El pipeline incorpora controles de calidad durante las transformaciones Raw → Silver → Gold.

Se comprobarán, entre otros:

- Valores nulos.
- Registros duplicados.
- Tipos y formatos.
- Homogeneidad de códigos municipales.
- Cobertura de municipios entre fuentes.
- Coherencia de las variables utilizadas en el análisis.
- Coordenadas y datos geográficos.
- Calidad del dataset final utilizado para scoring y clustering.

Los resultados concretos de estos controles se documentarán en la **tabla de calidad definida en el apartado 7.4**.

---

## 5. Solución y resultados

La solución utiliza un pipeline de datos basado en tres capas:

```text
INE / DGT / DGT NAP / CNIG
            │
            ▼
       ingestion.py
            │
            ▼
         S3 / RAW
            │
            ▼
        silver.py
            │
            ▼
       S3 / SILVER
            │
            ▼
         gold.py
            │
            ▼
        S3 / GOLD
            │
       ┌────┴─────┐
       ▼          ▼
 Scoring / ML  Streamlit
```

- **Raw:** datos originales obtenidos de las fuentes.
- **Silver:** limpieza, tipado, homogeneización, integración y tratamiento geoespacial.
- **Gold:** controles finales de calidad, variables derivadas, normalización y dataset analítico.

El análisis combinará:

1. **Baseline:** TODO - definir método de referencia.
2. **Scoring:** normalización, ponderación y ranking de municipios.
3. **K-Means:** segmentación de municipios con características similares.
4. **Streamlit:** visualización mediante mapa, KPIs, ranking, filtros y resultados.

### Resultados

TODO: completar con los resultados reales:

- Municipios con mayor score.
- Comparación con el baseline.
- Clusters identificados.
- Principales patrones encontrados.
- Visualizaciones relevantes.

---

## 6. Valor de negocio

El proyecto busca transformar diferentes fuentes públicas en una herramienta que facilite la **priorización de ubicaciones para infraestructura de recarga rápida**.

La recomendación final deberá indicar:

- Qué municipios deberían priorizarse.
- Por qué presentan mayor potencial.
- Qué variables explican su posición.
- Qué diferencias existen respecto al baseline.
- Qué impacto tendría seguir la recomendación.

**TODO:** incorporar la recomendación final, el impacto cuantificado y los supuestos definidos en el apartado 7.4 una vez obtenidos los resultados.

---

## 7. Limitaciones

Las principales limitaciones a evaluar son:

- Diferentes fechas de actualización entre fuentes.
- Posibles diferencias de cobertura y granularidad.
- Municipios o variables sin información completa.
- Dependencia del scoring respecto a variables, normalización y pesos seleccionados.
- Dependencia de K-Means respecto a las variables y número de clusters utilizados.
- Variables potencialmente relevantes no disponibles en las fuentes actuales.

Como siguientes pasos se podrá evaluar la incorporación de nuevas fuentes, variables de movilidad y tráfico, variables socioeconómicas y actualizaciones automáticas del pipeline.

---

## 8. Ejecución y uso de IA

### Estructura

```text
sot-01-ev-location/
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.yaml
├── src/
│   ├── gold.py
│   ├── ingestion.py
│   ├── orchestrator.py
│   └── silver.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

### Instalación

```bash
git clone <repository-url>
cd sot-01-ev-location

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Ejecución del pipeline

```bash
python3 src/orchestrator.py
```

`orchestrator.py` ejecuta secuencialmente:

```text
ingestion → Raw
silver    → Silver
gold      → Gold
```

La ejecución en Amazon EC2 utiliza el IAM Role asociado a la instancia para acceder a Amazon S3.

### Ejecución de Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Uso de IA generativa

Durante el desarrollo se ha utilizado IA generativa como herramienta de apoyo.

TODO: completar la declaración final indicando las herramientas utilizadas y las actividades concretas en las que se emplearon, de acuerdo con los requisitos de la entrega.
