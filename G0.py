import nltk
import psycopg2
from psycopg2 import sql
from datasets import load_dataset
import time
import statistics

nltk.download('punkt')

def download_dataset():
    dataset = load_dataset("bookcorpus", split='train', trust_remote_code=True)
    return dataset['text']

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def generate_chunks(texts, chunk_size=10000):
    sentences = []
    chunk_list = []

    for text in texts:
        sentences.extend(split_into_sentences(text))
        if len(sentences) >= chunk_size:
            chunk_list.append(sentences[:chunk_size])
            sentences = sentences[chunk_size:]

    if sentences:
        chunk_list.append(sentences)

    return chunk_list

def create_table(conn):
    create_table_query = """
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE IF NOT EXISTS dataset_chunks_pgvector (
        id SERIAL PRIMARY KEY,
        chunk_id INT,
        sentence TEXT,
        embedding vector(384)
    );
    """
    cur = conn.cursor()
    cur.execute(create_table_query)
    conn.commit()
    cur.close()

def insert_sentences(conn, chunk_id, sentences):
    insert_query = sql.SQL("INSERT INTO dataset_chunks_pgvector (chunk_id, sentence) VALUES (%s, %s)")
    cur = conn.cursor()

    start_time = time.time()
    print(f"Inserting sentences for chunk {chunk_id}...")

    for idx, sentence in enumerate(sentences):
        cur.execute(insert_query, (chunk_id, sentence))

    conn.commit()
    end_time = time.time()

    cur.close()
    print(f"Finished inserting sentences for chunk {chunk_id}. Total time: {end_time - start_time:.4f} seconds")
    return end_time - start_time

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

    create_table(conn)

    print("Downloading dataset...")
    max_texts = 50000
    texts = download_dataset()[:max_texts]

    print("Splitting dataset into chunks...")
    chunks = generate_chunks(texts, chunk_size=10000)

    insertion_times = []

    print("Inserting chunks into PostgreSQL...")
    for idx, chunk in enumerate(chunks):
        print(f"Inserting chunk {idx + 1}/{len(chunks)}")
        insertion_time = insert_sentences(conn, idx, chunk)
        insertion_times.append(insertion_time)

    if insertion_times:
        min_time = min(insertion_times)
        max_time = max(insertion_times)
        avg_time = statistics.mean(insertion_times)
        std_dev_time = statistics.stdev(insertion_times)

        print(f"Minimum insertion time: {min_time:.4f} seconds")
        print(f"Maximum insertion time: {max_time:.4f} seconds")
        print(f"Average insertion time: {avg_time:.4f} seconds")
        print(f"Standard deviation of insertion time: {std_dev_time:.4f} seconds")

    conn.close()
    print("Data inserted successfully!")

if __name__ == "__main__":
    main()
