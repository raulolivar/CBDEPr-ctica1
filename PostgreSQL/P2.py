import psycopg2
import numpy as np
import time
import statistics
from scipy.spatial.distance import euclidean, cityblock

def connect_to_postgres():
    conn = psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost',
    )
    return conn

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

    sentences = [(row[0], row[1]) for row in results]
    embeddings = [np.array(row[2]) for row in results]

    return sentences, embeddings

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

    all_sentences = [(row[0], row[1]) for row in results]
    all_embeddings = [np.array(row[2]) for row in results]

    return all_sentences, all_embeddings

def find_most_similar_sentences(sentence_id, sentence_embedding, embeddings, all_sentences, metric):
    distances = []

    for idx, emb in enumerate(embeddings):
        if all_sentences[idx][0] != sentence_id:
            if metric == 'euclidean':
                dist = euclidean(sentence_embedding, emb)
            elif metric == 'manhattan':
                dist = cityblock(sentence_embedding, emb)
            distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])

    top_2_similar = distances[:2]

    return [(all_sentences[idx], dist) for idx, dist in top_2_similar]

def process_sentences():
    conn = connect_to_postgres()

    first_10_sentences, first_10_embeddings = get_first_10_sentences_and_embeddings(conn)

    all_sentences, all_embeddings = get_all_sentences_and_embeddings(conn)

    print(f"\nComparing the first 10 sentences in chunk 0 with all sentences...")

    times = []

    for i, (sentence_id, sentence) in enumerate(first_10_sentences):
        print(f"\nSentence {i + 1}: {sentence}")

        sentence_embedding = first_10_embeddings[i]

        start_time = time.time()

        top_2_euclidean = find_most_similar_sentences(sentence_id, sentence_embedding, all_embeddings, all_sentences,
                                                      metric='euclidean')
        print("\nTop 2 most similar sentences (Euclidean Distance):")
        for (similar_sentence, dist) in top_2_euclidean:
            print(f"  - {similar_sentence[1]} (Distance: {dist:.4f})")

        top_2_manhattan = find_most_similar_sentences(sentence_id, sentence_embedding, all_embeddings, all_sentences,
                                                      metric='manhattan')
        print("\nTop 2 most similar sentences (Manhattan Distance):")
        for (similar_sentence, dist) in top_2_manhattan:
            print(f"  - {similar_sentence[1]} (Distance: {dist:.4f})")

        end_time = time.time()
        elapsed_time = end_time - start_time

        times.append(elapsed_time)

    conn.close()

    min_time = min(times)
    max_time = max(times)
    avg_time = statistics.mean(times)
    std_dev_time = statistics.stdev(times)

    print(f"\nTime Statistics:")
    print(f"  Minimum time: {min_time:.4f} seconds")
    print(f"  Maximum time: {max_time:.4f} seconds")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Standard deviation of time: {std_dev_time:.4f} seconds")

if __name__ == "__main__":
    process_sentences()
