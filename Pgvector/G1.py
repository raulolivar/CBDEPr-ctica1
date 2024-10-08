import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import statistics

def generate_embeddings(sentences, model):
    return model.encode(sentences)

def update_with_embeddings(conn, sentence_id, embedding):
    update_query = """
    UPDATE dataset_chunks_pgvector
    SET embedding = %s
    WHERE id = %s
    """
    cur = conn.cursor()
    cur.execute(update_query, (embedding.tolist(), sentence_id))
    conn.commit()
    cur.close()

def store_embeddings(conn):
    select_query = "SELECT id, sentence FROM dataset_chunks_pgvector WHERE embedding IS NULL AND chunk_id = 0"
    cur = conn.cursor()
    cur.execute(select_query)
    rows = cur.fetchall()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    times = []

    for row in rows:
        sentence_id, sentence = row
        start_time = time.time()
        embedding = generate_embeddings([sentence], model)[0]
        update_with_embeddings(conn, sentence_id, embedding)
        end_time = time.time()
        times.append(end_time - start_time)

    if times:
        print("Time statistics for storing embeddings:")
        print(
            f"Min: {min(times):.4f} sec, Max: {max(times):.4f} sec, Avg: {statistics.mean(times):.4f} sec, Std Dev: {statistics.stdev(times):.4f} sec")

def connect_to_postgres():
    conn = psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost',
    )
    return conn

def main():
    conn = connect_to_postgres()
    store_embeddings(conn)
    conn.close()

if __name__ == "__main__":
    main()
