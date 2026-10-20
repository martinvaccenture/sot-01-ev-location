# EV Location Intelligence

## 1. Descripción y objetivo

**EV Location Intelligence** es un proyecto de análisis de datos orientado a identificar y priorizar municipios españoles con mayor potencial para la instalación de hubs de recarga rápida de vehículos eléctricos.

El proyecto integra diferentes fuentes de datos públicas para construir un dataset municipal consolidado, generar un sistema de **scoring**, aplicar técnicas de **Machine Learning** y visualizar los resultados mediante una aplicación desarrollada con **Streamlit**.

### Objetivos

- Integrar información municipal procedente de diferentes fuentes.
- Construir un pipeline de datos basado en las capas **Raw → Silver → Gold**.
- Generar indicadores relevantes para evaluar los municipios.
- Diseñar un sistema de scoring para priorizar posibles ubicaciones.
- Aplicar **K-Means** para identificar grupos de municipios con características similares.
- Visualizar y comunicar los resultados mediante Streamlit.

---

## 2. Fuentes de datos

| Fuente | Información |
|---|---|
| **INE** | Información demográfica y municipal |
| **DGT** | Parque de vehículos e información relacionada con vehículos eléctricos |
| **DGT NAP** | Infraestructura y puntos de recarga, incluyendo puntos de potencia ≥ 50 kW |
| **CNIG** | Información geográfica y territorial |

La integración de las diferentes fuentes se realizará mediante identificadores territoriales comunes, principalmente el **código INE municipal**.

---

## 3. Arquitectura y pipeline

El proyecto sigue una arquitectura de procesamiento organizada en tres capas:

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

### Raw

Contiene los datos originales obtenidos desde las distintas fuentes, conservando la información antes de realizar transformaciones.

### Silver

Contiene los datos limpios y transformados. En esta capa se realizan procesos como:

- Tipado y estandarización.
- Tratamiento de valores nulos.
- Eliminación de duplicados.
- Homogeneización de códigos municipales.
- Integración entre fuentes.
- Tratamiento geoespacial.

### Gold

Contiene los datos preparados para su consumo analítico:

- Controles de calidad.
- Creación de variables derivadas.
- Normalización.
- Generación de indicadores.
- Preparación del scoring.
- Preparación de variables para Machine Learning.

El pipeline completo es coordinado por `orchestrator.py`, encargado de ejecutar secuencialmente las distintas fases.

---

## 4. Scoring y Machine Learning

A partir del dataset Gold se desarrollará un sistema de scoring para establecer un ranking de municipios:

```text
Variables
   │
   ▼
Normalización
   │
   ▼
Pesos
   │
   ▼
Score final
   │
   ▼
Ranking de municipios
```

Además, se utilizará **K-Means** para identificar grupos de municipios con características similares y comparar los resultados del clustering con la priorización obtenida mediante scoring.

**Pendiente de definir durante el desarrollo:**

- Variables definitivas.
- Método de normalización.
- Pesos utilizados.
- Análisis de sensibilidad.
- Número óptimo de clusters.
- Interpretación de los resultados.

---

## 5. Estructura del repositorio

```text
sot-01-ev-location/
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── gold.py
│   ├── ingestion.py
│   ├── orchestrator.py
│   └── silver.py
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Componentes principales

- **`src/ingestion.py`**: obtiene los datos de las distintas fuentes y los almacena en la capa Raw de Amazon S3.
- **`src/silver.py`**: realiza la limpieza, transformación, estandarización e integración de los datos.
- **`src/gold.py`**: prepara el dataset analítico final, realiza controles de calidad y genera las variables necesarias para scoring y Machine Learning.
- **`src/orchestrator.py`**: punto de entrada del pipeline. Ejecuta secuencialmente las fases de ingestión, Silver y Gold.
- **`app/streamlit_app.py`**: aplicación interactiva para visualizar los resultados finales.
- **`config/config.yaml`**: centraliza la configuración utilizada por los diferentes componentes.
- **`tests/`**: contiene las pruebas del pipeline y sus transformaciones.
- **`requirements.txt`**: contiene las dependencias Python necesarias para ejecutar el proyecto.
- **`.gitignore`**: excluye del repositorio credenciales, entornos virtuales, claves SSH, datasets locales y otros archivos que no deben versionarse.

---

## 6. Infraestructura y ejecución

El proyecto utiliza:

- **Amazon S3** para almacenar las capas Raw, Silver y Gold.
- **Amazon EC2** como entorno de ejecución.
- **AWS IAM** para gestionar los permisos entre EC2 y S3.
- **GitHub Enterprise** para control de versiones y colaboración.

El desarrollo se realiza en local utilizando ramas independientes. Una vez integrados los cambios en `main`, el repositorio se actualiza en EC2 para ejecutar el pipeline sobre AWS.

### Instalación

```bash
git clone <repository-url>
cd sot-01-ev-location

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Ejecución del pipeline

Desde la raíz del repositorio:

```bash
python3 src/orchestrator.py
```

El orquestador ejecuta:

```text
orchestrator.py
      │
      ├──► ingestion → S3 / Raw
      │
      ├──► silver    → S3 / Silver
      │
      └──► gold      → S3 / Gold
```

La instancia EC2 utiliza un **IAM Role** para acceder a los recursos necesarios de AWS sin almacenar credenciales directamente en el código.

### Ejecución del dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 7. Flujo de desarrollo

El desarrollo se organiza mediante ramas independientes por funcionalidad:

```text
feature/ingestion
feature/silver
feature/gold
feature/scoring
feature/streamlit
feature/orchestrator
        │
        ▼
   Pull Request
        │
        ▼
       main
        │
        ▼
       EC2
```

Cada funcionalidad se desarrolla de forma independiente para permitir el trabajo en paralelo.

Una vez finalizada y validada una funcionalidad, se crea una **Pull Request** hacia `main`. La versión integrada en `main` será la utilizada para ejecutar el proyecto en EC2.

---

## 8. Resultados

> Esta sección se completará una vez finalizado el procesamiento y análisis de los datos.

Los resultados finales incluirán:

- Ranking de municipios.
- Municipios prioritarios para nuevos hubs de recarga.
- Scores obtenidos.
- Clusters identificados.
- Principales indicadores.
- Visualizaciones del dashboard.
- Comparación entre scoring y clustering.
- Conclusiones y recomendación final.
