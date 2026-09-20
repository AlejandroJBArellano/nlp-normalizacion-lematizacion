"""
Pipeline de Normalizacion y Lematizacion de Texto en Espanol
Texto de prueba: Don Quijote de la Mancha (Miguel de Cervantes)
Checkpoint 2 + Checkpoint 4: incluye vectorizacion BoW y TF-IDF
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin display para entornos de terminal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import spacy
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA


def cargar_modelo_spacy(nombre_modelo="es_core_news_sm"):
    """Carga el modelo de spaCy en espanol o lo descarga si no existe."""
    try:
        nlp = spacy.load(nombre_modelo)
    except OSError:
        print(f"Descargando modelo {nombre_modelo}...")
        from spacy.cli import download
        download(nombre_modelo)
        nlp = spacy.load(nombre_modelo)
    return nlp


def cargar_texto(ruta_archivo="don_quijote.txt", solo_capitulo_1=True):
    """Carga el archivo de texto y opcionalmente extrae el Capitulo 1."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontro el archivo: {ruta_archivo}")

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    if solo_capitulo_1:
        inicio_patron = r"Capítulo primero\..*?(?=Capítulo II\.|\Z)"
        match = re.search(inicio_patron, contenido, re.DOTALL | re.IGNORECASE)
        if match:
            texto = match.group(0).strip()
            print(f"Capitulo 1 extraido exitosamente ({len(texto)} caracteres).")
            return texto

    print(f"Texto completo cargado ({len(contenido)} caracteres).")
    return contenido


def ejecutar_limpieza(texto, nlp):
    """Ejecuta el pipeline de tokenizacion, filtrado y lematizacion."""
    print("\n--- 1. Tokenizacion ---")
    doc = nlp(texto)
    print(f"Total de tokens iniciales: {len(doc)}")
    primeros_tokens = [token.text for token in doc if not token.is_space][:15]
    print(f"Primeros 15 tokens: {primeros_tokens}")

    print("\n--- 2. Filtrado de Stop Words y Puntuacion ---")
    tokens_relevantes = []
    tokens_ruido = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.strip():
            tokens_relevantes.append(token.text)
        elif token.is_stop or token.is_punct:
            tokens_ruido.append(token.text)

    print(f"Tokens eliminados (Ruido/Stopwords): {len(tokens_ruido)}")
    print(f"Tokens conservados (Contenido util): {len(tokens_relevantes)}")
    print(f"Muestra de palabras eliminadas: {tokens_ruido[:10]}")
    print(f"Muestra de palabras conservadas: {tokens_relevantes[:10]}")

    print("\n--- 3. Lematizacion y Normalizacion ---")
    tokens_normalizados = []
    cambios_interesantes = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.strip():
            lema = token.lemma_.lower()
            tokens_normalizados.append(lema)

            if token.text.lower() != lema and len(cambios_interesantes) < 10:
                cambios_interesantes.append(f"{token.text} -> {lema}")

    print(f"Total de tokens normalizados: {len(tokens_normalizados)}")
    print("Ejemplos de transformaciones morfologicas:")
    for cambio in cambios_interesantes:
        print(f"  * {cambio}")

    print(f"\nMuestra de tokens normalizados: {tokens_normalizados[:10]}")

    print("\n--- 4. Comparativa: Stemming vs Lematizacion ---")
    stemmer = SnowballStemmer("spanish")
    data_comparativa = []

    for token in doc:
        if not token.is_punct and not token.is_space and not token.is_stop and token.text.strip():
            raiz_stem = stemmer.stem(token.text)
            lema = token.lemma_.lower()
            data_comparativa.append({
                "Original": token.text,
                "Stemming (NLTK)": raiz_stem,
                "Lematizacion (spaCy)": lema,
                "Coinciden": raiz_stem == lema
            })

    df = pd.DataFrame(data_comparativa)

    palabras_clave = [
        "acordarme", "vivía", "antigua", "corredor", "leyendo",
        "imaginación", "deseaba", "caballeros", "hicieron", "podía"
    ]
    filtro = df[df["Original"].str.lower().isin(palabras_clave)].drop_duplicates(subset=["Original"])
    print("\nComparativa en palabras seleccionadas:")
    print(filtro.to_string(index=False))

    print("\nPrimeros 10 registros procesados:")
    print(df.head(10).to_string(index=False))

    print("\n--- 5. Reduccion de Dimensionalidad ---")
    vocabulario_original = len(set([t.text.lower() for t in doc if not t.is_punct and not t.is_space]))
    vocabulario_lemas = len(set(tokens_normalizados))
    reduccion_pct = ((vocabulario_original - vocabulario_lemas) / vocabulario_original) * 100

    print(f"Vocabulario unico original: {vocabulario_original}")
    print(f"Vocabulario unico lematizado: {vocabulario_lemas}")
    print(f"Reduccion de dimensionalidad: {reduccion_pct:.2f}%")

    return {
        "doc": doc,
        "tokens_normalizados": tokens_normalizados,
        "dataframe_comparativo": df
    }


def ejecutar_vectorizacion(doc):
    """Construye corpus lematizado por oraciones y aplica BoW y TF-IDF."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: VECTORIZACION DE TEXTO")
    print("=" * 60)

    # --- Corpus lematizado por oraciones ---
    print("\n--- 6. Corpus lematizado por oraciones ---")
    corpus_lematizado = []
    for oracion in doc.sents:
        lemas_oracion = [
            token.lemma_.lower()
            for token in oracion
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        if lemas_oracion:
            corpus_lematizado.append(" ".join(lemas_oracion))

    print(f"Total de oraciones en el corpus: {len(corpus_lematizado)}")
    print(f"Primera oracion del corpus: '{corpus_lematizado[0]}'")

    # --- Bag of Words ---
    print("\n--- 7a. Bag-of-Words (CountVectorizer) ---")
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
    print(f"Forma de la matriz BoW: {X_bow.shape}  (oraciones x terminos)")
    print(f"Vocabulario: {len(bow_vectorizer.vocabulary_)} terminos unicos")
    print(f"Densidad (no-ceros / total): "
          f"{X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]):.4f}")

    # --- TF-IDF ---
    print("\n--- 7b. TF-IDF (TfidfVectorizer) ---")
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)
    print(f"Forma de la matriz TF-IDF: {X_tfidf.shape}  (oraciones x terminos)")

    vocab_tfidf = tfidf_vectorizer.get_feature_names_out()
    primera = X_tfidf[0].toarray()[0]
    top_indices = np.argsort(primera)[::-1][:10]
    print("\nTop 10 terminos TF-IDF (primera oracion):")
    for i in top_indices:
        if primera[i] > 0:
            print(f"  {vocab_tfidf[i]:<20} peso: {primera[i]:.4f}")

    # --- Visualizacion 3D con PCA ---
    print("\n--- 8. Visualizacion 3D con PCA ---")
    _graficar_y_guardar(X_bow, bow_vectorizer.get_feature_names_out(),
                        X_tfidf, vocab_tfidf)

    return corpus_lematizado, X_bow, X_tfidf


def _graficar_y_guardar(X_bow, vocab_bow, X_tfidf, vocab_tfidf):
    """Genera el grafico 3D comparativo BoW vs TF-IDF y lo guarda como PNG."""

    def _plot_3d(ax, matriz, vocabulario, titulo, color):
        matriz_palabras = matriz.T
        pca = PCA(n_components=3)
        coords = pca.fit_transform(matriz_palabras.toarray())
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.scatter(x, y, z, c=color, s=80, edgecolors='k', alpha=0.8, depthshade=True)
        for i, palabra in enumerate(vocabulario[:30]):
            ax.text(x[i], y[i], z[i] + 0.01, palabra, fontsize=7)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel('CP1')
        ax.set_ylabel('CP2')
        ax.set_zlabel('CP3')
        ax.plot([0, 0], [0, 0], [z.min(), z.max()], c='grey', ls='--', lw=0.5, alpha=0.3)
        ax.plot([x.min(), x.max()], [0, 0], [0, 0], c='grey', ls='--', lw=0.5, alpha=0.3)
        ax.plot([0, 0], [y.min(), y.max()], [0, 0], c='grey', ls='--', lw=0.5, alpha=0.3)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('Representacion Vectorial del Quijote - Cap. 1', fontsize=14)
    ax1 = fig.add_subplot(121, projection='3d')
    _plot_3d(ax1, X_bow, vocab_bow, 'Espacio BoW 3D (Conteos)', 'orange')
    ax2 = fig.add_subplot(122, projection='3d')
    _plot_3d(ax2, X_tfidf, vocab_tfidf, 'Espacio TF-IDF 3D (Importancia)', 'teal')
    plt.tight_layout()

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "espacio_vectorial_3d.png")
    plt.savefig(salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura guardada en: {salida}")


def main():
    print("Iniciando pipeline de procesamiento de texto...")
    nlp = cargar_modelo_spacy("es_core_news_sm")
    ruta_libro = os.path.join(os.path.dirname(os.path.abspath(__file__)), "don_quijote.txt")
    texto = cargar_texto(ruta_libro, solo_capitulo_1=True)
    resultado = ejecutar_limpieza(texto, nlp)
    ejecutar_vectorizacion(resultado["doc"])
    print("\nProceso completado exitosamente.")


if __name__ == "__main__":
    main()
