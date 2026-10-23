# EV Location Intelligence

## 1. Problema y decisión

**EV Location Intelligence** busca apoyar la decisión de **en qué municipios españoles priorizar la instalación de nuevos hubs de recarga rápida para vehículos eléctricos**.

La decisión se apoya en datos demográficos, parque de vehículos, infraestructura de recarga existente e información geográfica, con el objetivo de identificar y ordenar municipios según su potencial.

El resultado del proyecto será una priorización de municipios que permita apoyar una decisión de localización basada en datos.

---

## 2. Hipótesis

Antes de realizar el análisis se plantean las siguientes hipótesis:

- H1: Mayor electrificación puede indicar mayor oportunidad.
- H2: Con demanda suficiente, menor cobertura puede indicar mayor oportunidad.
- H3: Crecimiento poblacional puede aumentar el potencial futuro.
- H4: El turismo puede representar demanda que el padrón no refleja.
- H5: Población por sí sola puede no capturar toda la oportunidad de inversión.

Estas hipótesis se contrastan mediante el scoring, la comparación con el baseline y el análisis de clustering.
---

## 3. Fuentes

| Fuente | Información utilizada | Fecha descarga | Licencia |
|---|---|---|---|
| INE | Información demográfica y municipal | 20/10/2026 | 	CC BY 4.0* |
| DGT | Parque de vehículos e información relacionada con vehículos eléctricos | 20/10/2026 | Aviso Legal DGT; no se identifica una licencia abierta específica asociada al fichero |
| DGT NAP | Infraestructura y puntos de recarga | 15/10/2026 | Aviso Legal DGT; no se identifica una licencia abierta específica asociada al fichero |
| CNIG | Información geográfica y territorial | Última actualización --> 28/07/2026 | CC BY 4.0 |

La integración de las fuentes se realizará utilizando identificadores territoriales comunes, principalmente el **código INE municipal**.

* Los conjuntos de datos de alto valor publicados por el INE están sujetos a licencia Creative Commons Attribution 4.0 (CC BY 4.0). La información de DGT se utiliza conforme a las condiciones establecidas en su Aviso Legal, al no haberse identificado una licencia abierta específica asociada al fichero descargado.

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
     ingestion_puntos_recarga.py
            │
            ▼
     ingestion_municipal.py
            │
            ▼
         S3 / RAW
            │
            ▼
     silver_cargadores_geometrias.py
            │
            ▼
     silver_municipal.py
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
            ▼
        Streamlit
```

- **Raw:** datos originales obtenidos de las fuentes.
- **Silver:** limpieza, tipado, homogeneización, integración y tratamiento geoespacial.
- **Gold:** controles finales de calidad, variables derivadas, normalización y dataset analítico.

El análisis combinará:

1. **Baseline:** Población y criterio del equipo de expansión. Baseline: seleccionar los 20 municipios con mayor población manteniendo la restricción de máximo 3 por provincia. 
2. **Scoring:** normalización, ponderación y ranking de municipios.
3. **K-Means:** segmentación de municipios con características similares.
4. **Streamlit:** visualización mediante mapa, KPIs, ranking, filtros y resultados.

### Resultados

El modelo genera un `opportunity_score` para cada municipio a partir de los indicadores normalizados definidos en la capa Gold. Este score permite construir un ranking de municipios según su oportunidad relativa para la instalación de nuevos hubs de recarga rápida.

#### Ranking de oportunidad

Los cinco municipios con mayor `opportunity_score` son:

| Posición | Municipio | Provincia | Población | Opportunity Score |
|---:|---|---|---:|---:|
| 1 | Moralzarzal | Madrid | 14.772 | 95,736 |
| 2 | Boadilla del Monte | Madrid | 66.349 | 94,860 |
| 3 | Valdemorillo | Madrid | 14.420 | 93,304 |
| 4 | Riba-roja de Túria | Valencia/València | 24.616 | 91,869 |
| 5 | Lliçà d'Amunt | Barcelona | 16.540 | 88,516 |

Los resultados muestran que los municipios con mayor oportunidad no son necesariamente los municipios con mayor población. El modelo destaca municipios de tamaño medio en los que la combinación de los indicadores utilizados genera una mayor puntuación relativa.

#### Comparación con el baseline

Como baseline se utiliza una priorización basada principalmente en población: se seleccionan los 20 municipios con mayor población, manteniendo el criterio de un máximo de 3 municipios por provincia.

La comparación entre ambos enfoques muestra diferencias significativas:

| Métrica | Modelo | Baseline |
|---|---:|---:|
| Coincidencias en el Top 20 | 0/20 | 0/20 |
| Opportunity Score medio de los municipios seleccionados | 85,04 | 43,68 |

No existe ninguna coincidencia entre los 20 municipios priorizados por el modelo y los seleccionados mediante el baseline. Además, los municipios seleccionados por el modelo presentan un `opportunity_score` medio de **85,04**, frente a **43,68** para los municipios seleccionados mediante el criterio de población.

Este resultado respalda la hipótesis **H5**, mostrando que la población por sí sola genera una priorización muy diferente a la obtenida mediante un enfoque multivariable. El modelo permite identificar oportunidades que no serían evidentes utilizando exclusivamente el tamaño de la población.

#### Segmentación mediante K-Means

Como análisis complementario se aplicó K-Means sobre las variables normalizadas para identificar grupos de municipios con características similares.

Se evaluaron valores de `K` entre 2 y 8 utilizando la inercia y el coeficiente de silueta:

| K | Inercia | Silueta |
|---:|---:|---:|
| 2 | 175,277 | 0,230 |
| 3 | 142,483 | 0,227 |
| 4 | 118,801 | 0,232 |
| 5 | 102,168 | 0,252 |
| **6** | **91,750** | **0,262** |
| 7 | 86,025 | 0,233 |
| 8 | 79,724 | 0,237 |

Se seleccionó **K = 6**, al presentar el mayor coeficiente de silueta entre las alternativas evaluadas (`0,262`), manteniendo al mismo tiempo una reducción de la inercia.

El clustering complementa el `opportunity_score` permitiendo identificar grupos de municipios con perfiles similares, mientras que el scoring proporciona la priorización final utilizada para apoyar la decisión de localización.

---

## 6. Valor de negocio

EV Location Intelligence transforma distintas fuentes públicas en una herramienta de apoyo a la decisión para priorizar municipios en los que estudiar la instalación de nuevos hubs de recarga rápida.

El principal valor de la solución es reducir un universo amplio de municipios a una lista priorizada y explicable, utilizando conjuntamente indicadores de electrificación, cobertura de recarga y evolución demográfica.

La comparación con el baseline muestra que el enfoque multivariable identifica oportunidades diferentes a las obtenidas utilizando principalmente la población:

- **0 coincidencias** entre los Top 20 del modelo y los Top 20 del baseline.
- Los municipios seleccionados por el modelo presentan un `opportunity_score` medio de **85,04**, frente a **43,68** en los municipios seleccionados por el baseline.
- El modelo permite identificar municipios de tamaño medio que quedarían fuera de una estrategia basada únicamente en población.

A partir de estos resultados, la recomendación es utilizar el `opportunity_score` como una primera fase de priorización para seleccionar los municipios que deben pasar a un análisis de viabilidad más detallado.

Los primeros municipios identificados son:

| Posición | Municipio | Provincia | Opportunity Score |
|---:|---|---|---:|
| 1 | Moralzarzal | Madrid | 95,736 |
| 2 | Boadilla del Monte | Madrid | 94,860 |
| 3 | Valdemorillo | Madrid | 93,304 |
| 4 | Riba-roja de Túria | Valencia/València | 91,869 |
| 5 | Lliçà d'Amunt | Barcelona | 88,516 |

La recomendación no implica instalar directamente un hub en estos municipios. El ranking permite priorizar dónde realizar las siguientes fases de análisis, incorporando información adicional como tráfico real, disponibilidad y capacidad de la red eléctrica, localización concreta, disponibilidad de suelo, competencia existente, costes y rentabilidad esperada.

De esta forma, la solución aporta valor al proceso de expansión al:

- **Focalizar el análisis** en los municipios con mayor oportunidad relativa.
- **Reducir el esfuerzo de análisis inicial**, descartando ubicaciones menos prioritarias.
- **Incorporar múltiples dimensiones** frente a una decisión basada únicamente en población.
- **Facilitar decisiones explicables**, mostrando los indicadores que contribuyen al score de cada municipio.
- **Permitir actualizar la priorización** cuando se incorporen nuevos datos mediante un pipeline reproducible.

---

## 7. Limitaciones

Las principales limitaciones a evaluar son:

- L1: Determinadas tablas DGT no cubren adecuadamente municipios pequeños. El universo de análisis queda restringido.
- L2: Los vehículos se atribuyen al domicilio del titular. Parque registrado no equivale necesariamente a movilidad real en el municipio.
- L3: El análisis de oportunidad no garantiza por sí solo la viabilidad técnica de una instalación. Un municipio atractivo puede requerir estudios adicionales antes de invertir.

Como siguientes pasos se podrá evaluar la incorporación de nuevas fuentes, variables de movilidad y tráfico, variables socioeconómicas y actualizaciones automáticas del pipeline.

---

## 8. Ejecución y uso de IA

### Estructura

```text
sot-01-ev-location/
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── bronze/
│   │   ├── ingesta_puntos_recarga.py
│   │   └── ingestion_municipal.py
│   │
│   ├── silver/
│   │   ├── silver_cargadores_geometria.py
│   │   └── silver_municipal.py
│   │
│   ├── gold/
│   │   ├── gold.py
│   │   └── pruebas_modelo.py
│   │
│   └── orchestrator.py
│
├── venv/
├── .env.example
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

## Uso de rutas S3

Las rutas S3 del proyecto se definen mediante variables de entorno en un archivo `.env`:

```env
AWS_REGION=eu-north-1
S3_RAW_PATH=s3://nombre-bucket/raw-data/
S3_SILVER_PATH=s3://nombre-bucket/silver/
S3_GOLD_PATH=s3://nombre-bucket/gold/
```

El archivo `.env` no se versiona en Git. Utilizar `.env.example` como plantilla para configurar el entorno local.

Para utilizar las rutas desde Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = os.getenv("S3_RAW_PATH")
SILVER_PATH = os.getenv("S3_SILVER_PATH")
GOLD_PATH = os.getenv("S3_GOLD_PATH")
```

Por ejemplo:

```python
input_path = f"{SILVER_PATH}municipios.parquet"
output_path = f"{GOLD_PATH}scoring_municipios.parquet"
```

Esto evita hardcodear las rutas S3 en los scripts y permite cambiar la configuración sin modificar el código.

### Uso de IA generativa

Durante el desarrollo se ha utilizado IA generativa como herramienta de apoyo.