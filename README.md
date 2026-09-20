# Proyecto PLN: Normalización, Lematización y Vectorización de Texto

Proyecto de Procesamiento de Lenguaje Natural que implementa el pipeline completo de preprocesamiento y representación vectorial sobre texto clásico en español (*Don Quijote de la Mancha* de Miguel de Cervantes).

## Contenido del Repositorio

- `don_quijote.txt`: Texto fuente del libro procesado.
- `limpieza_texto.ipynb`: Jupyter Notebook con el desarrollo paso a paso, visualizaciones y tablas comparativas.
- `limpieza_texto.py`: Script ejecutable en Python que implementa el pipeline completo.
- `requirements.txt`: Dependencias del proyecto congeladas para reproducibilidad.
- `espacio_vectorial_3d.png`: Gráfico generado automáticamente al ejecutar el script (espacio BoW y TF-IDF en 3D con PCA).
- `README.md`: Documentación y explicación técnica.

## Requisitos y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/AlejandroJBArellano/nlp-normalizacion-lematizacion.git
cd nlp-normalizacion-lematizacion
```

### 2. Crear y activar el entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate  # En Windows
```

### 3. Instalar dependencias y modelo lingüístico
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Ejecución

### Ejecución del Script Python
```bash
python limpieza_texto.py
```

### Ejecución del Cuaderno Jupyter
```bash
jupyter notebook limpieza_texto.ipynb
```

## Pipeline Completo

### Checkpoint 2 — Normalización y Lematización

1. **Tokenización**: Descomposición del flujo de texto en tokens individuales utilizando el modelo `es_core_news_sm` de spaCy.
2. **Filtrado de Ruido (Stop Words y Puntuación)**: Eliminación de palabras funcionales y caracteres no alfanuméricos que no aportan valor semántico directo.
3. **Lematización y Normalización**: Transformación de palabras flexionadas (verbos conjugados, plurales, géneros) a su forma canónica o lema en minúsculas.
4. **Comparativa: Stemming vs Lematización**:
   - **Stemming (NLTK Snowball)**: Algoritmo heurístico que corta sufijos de forma rápida pero sin validar la existencia de la raíz léxica en diccionario.
   - **Lematización (spaCy)**: Análisis morfológico informado por contexto lingüístico que asegura lemas válidos.
5. **Reducción de Dimensionalidad**: Reducción del vocabulario único en más del 30%, mitigando la dispersión de datos (*sparsity*).

### Checkpoint 4 — Representación Vectorial y Semántica (Feature Extraction)

6. **Corpus por oraciones**: Construcción de un corpus donde cada oración del capítulo es un documento, con sus lemas como contenido.
7. **Bag-of-Words (CountVectorizer)**: Representación vectorial basada en conteos de términos. Simple pero efectiva para clasificación de temas. Genera vectores dispersos.
8. **TF-IDF (TfidfVectorizer)**: Ponderación estadística que premia los términos frecuentes en un documento y penaliza los que aparecen en todo el corpus. Captura la *relevancia relativa* de cada término.
9. **Visualización en Espacio Vectorial 3D**: Reducción de dimensionalidad con PCA (3 componentes principales) y visualización comparativa de los espacios BoW y TF-IDF.

## Resumen de Resultados — Checkpoint 2

| Métrica | Valor |
|---|---|
| Tokens Iniciales | 2,329 |
| Tokens Útiles (Sin Ruido) | 723 |
| Tokens Eliminados (Stopwords/Puntuación) | 1,458 |
| Vocabulario Único Original | 719 |
| Vocabulario Único Lematizado | 489 |
| Reducción de Dimensionalidad | 31.99% |

## Resumen de Resultados — Checkpoint 4

| Modelo | Dimensión de la Matriz | Observación |
|---|---|---|
| Bag-of-Words | `oraciones × términos_únicos` | Conteos crudos, alta sparsity |
| TF-IDF | `oraciones × términos_únicos` | Pesos normalizados, mayor separación semántica |
| PCA 3D | 3 componentes principales | Permite visualizar el espacio vectorial |
