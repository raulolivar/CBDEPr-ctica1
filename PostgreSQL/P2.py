import psycopg2
import numpy as np
import time
import statistics
from scipy.spatial.distance import euclidean, cityblock

# Conexión a la base de datos PostgreSQL
def connect_to_postgres():
    conn = psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost',
    )
    return conn

# Obtener las 10 primeras oraciones del chunk 0 y sus embeddings
def get_first_10_sentences_and_embeddings(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT se.sentence_id, dc.sentence, se.sentence_embedding 
        FROM dataset_chunks dc
        JOIN sentence_embeddings se ON dc.id = se.sentence_id
        WHERE dc.chunk_id = 0 
        LIMIT 10
    """)
    results = cur.fetchall()
    cur.close()

    # Extraer las oraciones y embeddings
    sentences = [(row[0], row[1]) for row in results]
    embeddings = [np.array(row[2]) for row in results]

    return sentences, embeddings

# Obtener todas las oraciones y sus embeddings
def get_all_sentences_and_embeddings(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT se.sentence_id, dc.sentence, se.sentence_embedding 
        FROM dataset_chunks dc
        JOIN sentence_embeddings se ON dc.id = se.sentence_id
        WHERE dc.chunk_id = 0
    """)
    results = cur.fetchall()
    cur.close()

    # Extraer las oraciones y embeddings
    all_sentences = [(row[0], row[1]) for row in results]
    all_embeddings = [np.array(row[2]) for row in results]

    return all_sentences, all_embeddings

# Calcular las 2 frases más similares usando métricas Euclidean y Manhattan
def find_most_similar_sentences(sentence_id, sentence_embedding, embeddings, all_sentences, metric):
    distances = []

    # Calcular las distancias entre la oración actual y todas las demás
    for idx, emb in enumerate(embeddings):
        if all_sentences[idx][0] != sentence_id:  # No comparar la oración consigo misma
            if metric == 'euclidean':
                dist = euclidean(sentence_embedding, emb)
            elif metric == 'manhattan':
                dist = cityblock(sentence_embedding, emb)
            distances.append((idx, dist))

    # Ordenar las distancias en orden ascendente
    distances.sort(key=lambda x: x[1])

    # Obtener las 2 oraciones más similares
    top_2_similar = distances[:2]

    return [(all_sentences[idx], dist) for idx, dist in top_2_similar]

# Procesar las oraciones y calcular tiempos
def process_sentences():
    conn = connect_to_postgres()

    # Obtener las 10 primeras oraciones del chunk 0 y sus embeddings
    first_10_sentences, first_10_embeddings = get_first_10_sentences_and_embeddings(conn)

    # Obtener todas las oraciones y embeddings
    all_sentences, all_embeddings = get_all_sentences_and_embeddings(conn)

    print(f"\nComparing the first 10 sentences in chunk 0 with all sentences...")

    times = []  # Lista para almacenar los tiempos

    for i, (sentence_id, sentence) in enumerate(first_10_sentences):
        print(f"\nSentence {i + 1}: {sentence}")

        sentence_embedding = first_10_embeddings[i]

        # Medir el tiempo de inicio para Euclidean y Manhattan
        start_time = time.time()

        # Encontrar las 2 oraciones más similares usando la distancia euclidiana
        top_2_euclidean = find_most_similar_sentences(sentence_id, sentence_embedding, all_embeddings, all_sentences,
                                                      metric='euclidean')
        print("\nTop 2 most similar sentences (Euclidean Distance):")
        for (similar_sentence, dist) in top_2_euclidean:
            print(f"  - {similar_sentence[1]} (Distance: {dist:.4f})")

        # Encontrar las 2 oraciones más similares usando la distancia manhattan
        top_2_manhattan = find_most_similar_sentences(sentence_id, sentence_embedding, all_embeddings, all_sentences,
                                                      metric='manhattan')
        print("\nTop 2 most similar sentences (Manhattan Distance):")
        for (similar_sentence, dist) in top_2_manhattan:
            print(f"  - {similar_sentence[1]} (Distance: {dist:.4f})")

        # Medir el tiempo final
        end_time = time.time()
        elapsed_time = end_time - start_time

        times.append(elapsed_time)  # Almacenar el tiempo para este cálculo

    conn.close()

    # Calcular las estadísticas de tiempo
    min_time = min(times)
    max_time = max(times)
    avg_time = statistics.mean(times)
    std_dev_time = statistics.stdev(times)

    # Mostrar los resultados
    print(f"\nTime Statistics:")
    print(f"  Minimum time: {min_time:.4f} seconds")
    print(f"  Maximum time: {max_time:.4f} seconds")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Standard deviation of time: {std_dev_time:.4f} seconds")

if __name__ == "__main__":
    process_sentences()
