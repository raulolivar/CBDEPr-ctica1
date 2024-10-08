import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
import time
import statistics

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def connect_to_postgres():
    conn = psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost',
    )
    return conn

def create_embedding_table(conn):
    create_table_query = """
    CREATE TABLE sentence_embeddings (
        sentence_id INT,
        chunk_id INT,
        sentence_embedding FLOAT8[],
        PRIMARY KEY (sentence_id, chunk_id),
        CONSTRAINT fk_sentence
          FOREIGN KEY(sentence_id) 
          REFERENCES dataset_chunks(id)
          ON DELETE CASCADE
    );
    """
    cur = conn.cursor()
    cur.execute(create_table_query)
    conn.commit()
    cur.close()

def get_sentences_with_chunk_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_chunks")
    sentences = cur.fetchall()
    cur.close()
    return sentences

def insert_embeddings(conn, sentence_id, chunk_id, embedding):
    insert_query = sql.SQL(
        "INSERT INTO sentence_embeddings (sentence_id, chunk_id, sentence_embedding) VALUES (%s, %s, %s)")
    cur = conn.cursor()
    cur.execute(insert_query, (sentence_id, chunk_id, embedding))
    conn.commit()
    cur.close()

def process_embeddings(conn):
    sentences = get_sentences_with_chunk_ids(conn)
    print(f"Generating embeddings for {len(sentences)} sentences...")

    insertion_times = []

    start_time_total = time.time()

    for sentence_id, chunk_id, sentence in sentences:
        embedding_start_time = time.time()
        embedding = model.encode(sentence).tolist()

        insert_embeddings(conn, sentence_id, chunk_id, embedding)
        embedding_end_time = time.time()

        insertion_time = embedding_end_time - embedding_start_time
        insertion_times.append(insertion_time)

        print(f"Inserted embedding for sentence {sentence_id} in chunk {chunk_id} in {insertion_time:.4f} seconds")

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"Finished processing embeddings. Total time: {total_time:.4f} seconds")

    if insertion_times:
        min_time = min(insertion_times)
        max_time = max(insertion_times)
        avg_time = statistics.mean(insertion_times)
        std_dev_time = statistics.stdev(insertion_times)

        print(f"\n--- Embedding Insertion Time Statistics ---")
        print(f"Minimum time: {min_time:.4f} seconds")
        print(f"Maximum time: {max_time:.4f} seconds")
        print(f"Average time: {avg_time:.4f} seconds")
        print(f"Standard deviation: {std_dev_time:.4f} seconds")
    else:
        print("No insertion times recorded.")

def main():
    conn = connect_to_postgres()

    create_embedding_table(conn)

    process_embeddings(conn)

    conn.close()
    print("Embeddings inserted successfully!")

if __name__ == "__main__":
    main()
