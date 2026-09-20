"""
Pipeline completo de PLN: Normalizacion, Vectorizacion y Semantica Distribucional
Texto de prueba: Don Quijote de la Mancha (Miguel de Cervantes)
Checkpoints 2, 4 y 5
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import spacy
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


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
            print(f"Capitulo 1 extraido ({len(texto)} caracteres).")
            return texto, contenido

    print(f"Texto completo cargado ({len(contenido)} caracteres).")
    return contenido, contenido


def ejecutar_limpieza(texto, nlp):
    """Pipeline de tokenizacion, filtrado y lematizacion."""
    print("\n--- 1. Tokenizacion ---")
    doc = nlp(texto)
    print(f"Total de tokens iniciales: {len(doc)}")

    print("\n--- 2. Filtrado de Stop Words y Puntuacion ---")
    tokens_relevantes = []
    tokens_ruido = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.strip():
            tokens_relevantes.append(token.text)
        elif token.is_stop or token.is_punct:
            tokens_ruido.append(token.text)
    print(f"Tokens eliminados: {len(tokens_ruido)}  |  conservados: {len(tokens_relevantes)}")

    print("\n--- 3. Lematizacion y Normalizacion ---")
    tokens_normalizados = []
    cambios = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.strip():
            lema = token.lemma_.lower()
            tokens_normalizados.append(lema)
            if token.text.lower() != lema and len(cambios) < 10:
                cambios.append(f"{token.text} -> {lema}")
    print(f"Total tokens normalizados: {len(tokens_normalizados)}")
    for c in cambios:
        print(f"  * {c}")

    print("\n--- 4. Comparativa: Stemming vs Lematizacion ---")
    stemmer = SnowballStemmer("spanish")
    data = []
    for token in doc:
        if not token.is_punct and not token.is_space and not token.is_stop and token.text.strip():
            data.append({
                "Original": token.text,
                "Stemming (NLTK)": stemmer.stem(token.text),
                "Lematizacion (spaCy)": token.lemma_.lower(),
                "Coinciden": stemmer.stem(token.text) == token.lemma_.lower()
            })
    df = pd.DataFrame(data)
    print(df.head(10).to_string(index=False))

    print("\n--- 5. Reduccion de Dimensionalidad ---")
    vocab_orig = len(set([t.text.lower() for t in doc if not t.is_punct and not t.is_space]))
    vocab_lem = len(set(tokens_normalizados))
    reduccion = ((vocab_orig - vocab_lem) / vocab_orig) * 100
    print(f"Vocabulario original: {vocab_orig}  ->  lematizado: {vocab_lem}  ({reduccion:.2f}% reduccion)")

    return {"doc": doc, "tokens_normalizados": tokens_normalizados}


def ejecutar_vectorizacion(doc):
    """Construye corpus lematizado por oraciones y aplica BoW y TF-IDF."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: VECTORIZACION")
    print("=" * 60)

    corpus = []
    for oracion in doc.sents:
        lemas = [
            token.lemma_.lower()
            for token in oracion
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        if lemas:
            corpus.append(" ".join(lemas))

    print(f"Oraciones en el corpus: {len(corpus)}")

    bow_vec = CountVectorizer()
    X_bow = bow_vec.fit_transform(corpus)
    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(corpus)

    print(f"BoW:   {X_bow.shape}  |  TF-IDF: {X_tfidf.shape}")
    print(f"Densidad BoW: {X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]):.4f}")

    vocab_tfidf = tfidf_vec.get_feature_names_out()
    primera = X_tfidf[0].toarray()[0]
    top = np.argsort(primera)[::-1][:10]
    print("\nTop 10 terminos TF-IDF (primera oracion):")
    for i in top:
        if primera[i] > 0:
            print(f"  {vocab_tfidf[i]:<20} {primera[i]:.4f}")

    _graficar_bow_tfidf(X_bow, bow_vec.get_feature_names_out(),
                        X_tfidf, vocab_tfidf)
    return corpus, X_bow, X_tfidf, bow_vec, tfidf_vec


def _graficar_bow_tfidf(X_bow, vocab_bow, X_tfidf, vocab_tfidf):
    def _plot(ax, mat, voc, titulo, color):
        coords = PCA(n_components=3).fit_transform(mat.T.toarray())
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.scatter(x, y, z, c=color, s=80, edgecolors='k', alpha=0.8, depthshade=True)
        for i, p in enumerate(voc[:30]):
            ax.text(x[i], y[i], z[i] + 0.01, p, fontsize=7)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel('CP1'); ax.set_ylabel('CP2'); ax.set_zlabel('CP3')

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('Representacion Vectorial del Quijote - Cap. 1', fontsize=14)
    ax1 = fig.add_subplot(121, projection='3d')
    _plot(ax1, X_bow, vocab_bow, 'Espacio BoW 3D (Conteos)', 'orange')
    ax2 = fig.add_subplot(122, projection='3d')
    _plot(ax2, X_tfidf, vocab_tfidf, 'Espacio TF-IDF 3D (Importancia)', 'teal')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "espacio_vectorial_3d.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Guardado: {out}")


def ejecutar_semantica_distribucional(texto_completo, nlp):
    """Checkpoint 5: GloVe pre-entrenado y Word2Vec desde cero."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 5: SEMANTICA DISTRIBUCIONAL")
    print("=" * 60)

    # --- 9. GloVe pre-entrenado ---
    print("\n--- 9. Embeddings GloVe 50d ---")
    try:
        import gensim.downloader as api
        print("Cargando GloVe (primera vez descarga ~66 MB)...")
        glove = api.load("glove-wiki-gigaword-50")
        print(f"Vocabulario GloVe: {len(glove)} palabras")

        words = ["man", "woman", "boy", "girl", "king", "queen"]
        vecs = np.array([glove[w] for w in words])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(vecs, cmap="RdBu_r", aspect="auto")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=13)
        ax.set_xlabel("Dimension (1-50)")
        ax.set_title("Embeddings GloVe - vectores densos de 50 dimensiones")
        plt.colorbar(ax.images[0], ax=ax, label="Valor")
        plt.tight_layout()
        out_hm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove_heatmap.png")
        plt.savefig(out_hm, dpi=150)
        plt.close(fig)
        print(f"Guardado: {out_hm}")

        # --- 10. Aritmetica semantica ---
        print("\n--- 10. Aritmetica Semantica: king - man + woman ---")
        resultado = glove["king"] - glove["man"] + glove["woman"]
        sim = cosine_similarity([resultado], [glove["queen"]])[0][0]
        print(f"Similitud coseno (resultado vs queen): {sim:.4f}")

        top5 = glove.similar_by_vector(resultado, topn=5)
        print("Palabras mas cercanas al resultado:")
        for w, s in top5:
            print(f"  {w:<15} {s:.4f}")

        labels = ["king", "- man", "+ woman", "= resultado", "queen (real)"]
        vecs_an = np.array([glove["king"], -glove["man"], glove["woman"], resultado, glove["queen"]])
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(vecs_an, cmap="RdBu_r", aspect="auto")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dimension")
        ax.set_title("Analogia: king - man + woman = queen", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Valor")
        ax.annotate(f"similitud coseno = {sim:.4f}",
                    xy=(0.5, -0.22), xycoords="axes fraction",
                    ha="center", fontsize=12, fontstyle="italic")
        plt.tight_layout()
        out_an = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove_analogy.png")
        plt.savefig(out_an, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {out_an}")

    except Exception as e:
        print(f"Error cargando GloVe: {e}")
        print("Instala gensim: pip install gensim")

    # --- 11. Word2Vec desde cero ---
    print("\n--- 11. Word2Vec Skip-gram — Don Quijote completo ---")
    try:
        import multiprocessing
        from gensim.models import Word2Vec

        # Usar los primeros 500K chars del libro completo
        doc_completo = nlp(texto_completo[:500000])
        sentences_w2v = []
        for sent in doc_completo.sents:
            tokens = [
                token.lemma_.lower()
                for token in sent
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            if len(tokens) > 1:
                sentences_w2v.append(tokens)

        print(f"Oraciones de entrenamiento: {len(sentences_w2v)}")
        w2v = Word2Vec(
            sentences_w2v,
            vector_size=50,
            window=5,
            min_count=3,
            workers=multiprocessing.cpu_count(),
            sg=1,
            seed=42
        )
        print(f"Vocabulario Word2Vec: {len(w2v.wv)} palabras")

        # --- 12. Exploracion semantica ---
        print("\n--- 12. Exploracion Semantica ---")
        for palabra in ["caballero", "hidalgo", "espada", "batalla", "amor"]:
            try:
                sims = w2v.wv.most_similar(palabra, topn=5)
                print(f"\nMas cercanas a '{palabra}':")
                for w, s in sims:
                    print(f"  {w:<20} {s:.4f}")
            except KeyError:
                print(f"'{palabra}' no en vocabulario")

        # --- 13. Visualizacion 3D Word2Vec ---
        print("\n--- 13. Visualizacion 3D Word2Vec ---")
        vocab_w2v = list(w2v.wv.index_to_key)[:80]
        vectores = w2v.wv[vocab_w2v]
        coords = PCA(n_components=3).fit_transform(vectores)
        df_w2v = pd.DataFrame(coords, columns=["x", "y", "z"])
        df_w2v["palabra"] = vocab_w2v

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df_w2v["x"], df_w2v["y"], df_w2v["z"],
                   c="crimson", s=80, edgecolors="white", alpha=0.8)
        for _, row in df_w2v.head(30).iterrows():
            ax.text(row["x"], row["y"], row["z"], f" {row['palabra']}", size=9)
        ax.set_title("Espacio Semantico Word2Vec - Don Quijote", fontsize=14, fontweight="bold")
        ax.set_xlabel("Dimension Latente 1")
        ax.set_ylabel("Dimension Latente 2")
        ax.set_zlabel("Dimension Latente 3")
        plt.tight_layout()
        out_w2v = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_3d.png")
        plt.savefig(out_w2v, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {out_w2v}")

    except Exception as e:
        print(f"Error en Word2Vec: {e}")


def main():
    print("Iniciando pipeline de procesamiento de texto...")
    nlp = cargar_modelo_spacy("es_core_news_sm")
    ruta_libro = os.path.join(os.path.dirname(os.path.abspath(__file__)), "don_quijote.txt")
    texto_cap1, texto_completo = cargar_texto(ruta_libro, solo_capitulo_1=True)

    resultado = ejecutar_limpieza(texto_cap1, nlp)
    ejecutar_vectorizacion(resultado["doc"])
    ejecutar_semantica_distribucional(texto_completo, nlp)

    print("\nProceso completado exitosamente.")


if __name__ == "__main__":
    main()
