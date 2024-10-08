import psycopg2
import time
import statistics

def get_embeddings(conn):
    select_query = "SELECT id, sentence, embedding::vector FROM dataset_chunks_pgvector WHERE embedding IS NOT NULL AND id > 0 AND id < 11 LIMIT 10"
    cur = conn.cursor()
    cur.execute(select_query)
    rows = cur.fetchall()
    cur.close()
    return rows

def find_top2_similar(conn, target_id, metric='cosine'):
    if metric == 'cosine':
        metric_function = '<=>'  # Cosine distance
    elif metric == 'l2':
        metric_function = '<->'  # L2 distance

    query = f"""
    SELECT id, sentence
    FROM dataset_chunks_pgvector
    WHERE id != %s AND embedding IS NOT NULL
    ORDER BY embedding {metric_function} (SELECT embedding::vector FROM dataset_chunks_pgvector WHERE id = %s)
    LIMIT 2;
    """
    cur = conn.cursor()
    cur.execute(query, (target_id, target_id))
    rows = cur.fetchall()
    cur.close()
    return rows

def compare_sentences(conn):
    sentences_embeddings = get_embeddings(conn)
    times= []

    for target_id, target_sentence, target_embedding in sentences_embeddings:
        print(f"Comparing sentence: {target_sentence}")

        # cosine
        start_time = time.time()
        top2_cosine = find_top2_similar(conn, target_id, metric='cosine')
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Top 2 cosine similar: {top2_cosine}")

        # euclidean
        start_time = time.time()
        top2_l2 = find_top2_similar(conn, target_id, metric='l2')
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Top 2 L2 similar: {top2_l2}")

        print("Distance times:")
        print(
            f"Min: {min(times):.4f} sec, Max: {max(times):.4f} sec, Avg: {statistics.mean(times):.4f} sec, Std Dev: {statistics.stdev(times):.4f} sec"
        )

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
    compare_sentences(conn)
    conn.close()

if __name__ == "__main__":
    main()
