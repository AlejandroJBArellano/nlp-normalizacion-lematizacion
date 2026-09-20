"""
Pipeline de Normalizacion y Lematizacion de Texto en Espanol
Texto de prueba: Don Quijote de la Mancha (Miguel de Cervantes)
"""

import os
import re
import pandas as pd
import spacy
from nltk.stem import SnowballStemmer


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
    print(f"Ejemplos de transformaciones morfologicas:")
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


def main():
    print("Iniciando pipeline de procesamiento de texto...")
    nlp = cargar_modelo_spacy("es_core_news_sm")
    ruta_libro = os.path.join(os.path.dirname(__file__), "don_quijote.txt")
    texto = cargar_texto(ruta_libro, solo_capitulo_1=True)
    ejecutar_limpieza(texto, nlp)
    print("\nProceso completado exitosamente.")


if __name__ == "__main__":
    main()
