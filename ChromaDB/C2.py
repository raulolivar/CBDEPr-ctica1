import chromadb
import numpy as np
import time
import statistics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

client = chromadb.PersistentClient(path="/c/Users/Sergi Jaume/Desktop/CBDE/CBDEPr-ctica1")
collection_name = "text_data_collection"

metadata_cosine = {"hnsw:space": "cosine"}
metadata_l2 = {"hnsw:space": "l2"}

collection_cosine = client.get_collection(name=collection_name, metadata=metadata_cosine)
collection_l2 = client.get_collection(name=collection_name, metadata=metadata_l2)

embeddings = collection_cosine.get()['embeddings']
documents = collection_cosine.get()['documents']

query_sentences = documents[:10]

def find_top_k_similar(distances_matrix, top_k=2):
    similar_sentences = []
    for i, row in enumerate(distances_matrix):
        top_k_indices = np.argsort(row)[1:top_k + 1]
        similar_sentences.append([documents[idx] for idx in top_k_indices])
    return similar_sentences

times = []

for i, query_sentence in enumerate(query_sentences):
    current_embedding = np.array([embeddings[i]])

    start_time_cosine = time.time()
    cosine_distances_matrix = cosine_distances(current_embedding, embeddings)
    top_k_cosine_similar = find_top_k_similar(cosine_distances_matrix, top_k=2)
    end_time_cosine = time.time()
    times.append(end_time_cosine - start_time_cosine)

    start_time_l2 = time.time()
    l2_distances_matrix = euclidean_distances(current_embedding, embeddings)
    top_k_l2_similar = find_top_k_similar(l2_distances_matrix, top_k=2)
    end_time_l2 = time.time()
    times.append(end_time_l2 - start_time_l2)

    print(f"\nQuery sentence: {query_sentence}")
    print("Top-2 similar sentences (Cosine Distance):")
    for sim_sentence in top_k_cosine_similar:
        print(f"  - {sim_sentence}")

    print("Top-2 similar sentences (L2 Distance):")
    for sim_sentence in top_k_l2_similar:
        print(f"  - {sim_sentence}")

min_time = min(times)
max_time = max(times)
avg_time = statistics.mean(times)
std_dev = statistics.stdev(times)

print("\nL2 Distance Time Statistics:")
print(f"Minimum time: {min_time:.6f} seconds")
print(f"Maximum time: {max_time:.6f} seconds")
print(f"Average time: {avg_time:.6f} seconds")
print(f"Standard deviation: {std_dev:.6f} seconds")
