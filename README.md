# Proyecto PLN: Normalización, Vectorización y Semántica Distribucional

Proyecto de Procesamiento de Lenguaje Natural que implementa el pipeline completo sobre texto clásico en español (*Don Quijote de la Mancha* de Miguel de Cervantes).

## Contenido del Repositorio

| Archivo | Descripción |
|---|---|
| `don_quijote.txt` | Texto fuente del libro procesado |
| `limpieza_texto.ipynb` | Notebook Jupyter con el pipeline completo paso a paso |
| `limpieza_texto.py` | Script ejecutable en Python |
| `requirements.txt` | Dependencias congeladas para reproducibilidad |
| `espacio_vectorial_3d.png` | Espacio BoW y TF-IDF en 3D (PCA) |
| `glove_heatmap.png` | Heatmap de embeddings GloVe (50 dimensiones) |
| `glove_analogy.png` | Visualización de la analogía `king - man + woman ≈ queen` |
| `word2vec_3d.png` | Espacio semántico Word2Vec del Quijote en 3D |

## Instalación

```bash
git clone https://github.com/AlejandroJBArellano/nlp-normalizacion-lematizacion.git
cd nlp-normalizacion-lematizacion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Ejecución

```bash
# Script completo (genera todas las figuras PNG)
python limpieza_texto.py

# Notebook interactivo
jupyter notebook limpieza_texto.ipynb
```

## Pipeline Completo

### Checkpoint 2 — Normalización y Lematización

1. **Tokenización**: Descomposición del flujo de texto en tokens con el modelo `es_core_news_sm` de spaCy.
2. **Filtrado de Ruido**: Eliminación de stop words, puntuación y espacios.
3. **Lematización y Normalización**: Transformación de palabras flexionadas a lemas canónicos en minúsculas.
4. **Stemming vs Lematización**: Comparativa NLTK Snowball vs spaCy morfológico.
5. **Reducción de Dimensionalidad**: Contracción del vocabulario único en más del 30%.

### Checkpoint 4 — Representación Vectorial y Semántica

6. **Corpus por oraciones**: Cada oración del capítulo es un documento del corpus.
7. **Bag-of-Words** (`CountVectorizer`): Vectores de conteos, alta sparsity.
8. **TF-IDF** (`TfidfVectorizer`): Pesos normalizados, mayor separación semántica.
9. **Visualización 3D**: PCA sobre ambas matrices, gráfico comparativo lado a lado.

### Checkpoint 5 — Semántica Distribucional

10. **GloVe pre-entrenado** (`glove-wiki-gigaword-50`): Heatmap de vectores densos de 50d.
11. **Aritmética semántica**: `king - man + woman ≈ queen` (similitud coseno > 0.85).
12. **Word2Vec Skip-gram** entrenado sobre el Quijote completo (500K chars): vocabulario propio.
13. **Exploración semántica**: Palabras más cercanas a `caballero`, `hidalgo`, `espada`, etc.
14. **Espacio semántico 3D**: PCA sobre embeddings Word2Vec, top 80 palabras.

## Resultados

### Checkpoint 2

| Métrica | Valor |
|---|---|
| Tokens Iniciales | 2,329 |
| Tokens Útiles (Sin Ruido) | 723 |
| Vocabulario Único Original | 719 |
| Vocabulario Único Lematizado | 489 |
| Reducción de Dimensionalidad | 31.99% |

### Checkpoint 4

| Modelo | Dimensión | Observación |
|---|---|---|
| Bag-of-Words | `oraciones × términos` | Conteos crudos, alta sparsity |
| TF-IDF | `oraciones × términos` | Pesos normalizados |

### Checkpoint 5

| Modelo | Tipo | Observación |
|---|---|---|
| GloVe 50d | Pre-entrenado (inglés) | Vectores densos universales |
| Word2Vec Skip-gram | Propio (español) | Entrenado sobre Don Quijote |
| Analogía semántica | king-man+woman | Similitud coseno vs queen > 0.85 |
