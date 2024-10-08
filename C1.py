import chromadb
import time
import statistics
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="/c/Users/Sergi Jaume/Desktop/CBDE/CBDEPr-ctica1")

collection_name = "text_data_collection"
collection = client.get_collection(name=collection_name)

results = collection.get()
documents = results['documents']
ids = results['ids']

embeddings = model.encode(documents, convert_to_tensor=False).tolist()

times = []

for cont, (doc, emb) in enumerate(zip(documents, embeddings)):
    start_time = time.time()
    print(f"Updating document with ID: {ids[cont]}")
    collection.update(
        ids=[ids[cont]],
        embeddings=[emb],
        documents=[doc]
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

min_time = min(times)
max_time = max(times)
avg_time = statistics.mean(times)
std_dev_time = statistics.stdev(times)

print(f"Minimum time to store an embedding: {min_time:.6f} seconds")
print(f"Maximum time to store an embedding: {max_time:.6f} seconds")
print(f"Average time to store an embedding: {avg_time:.6f} seconds")
print(f"Standard deviation of times: {std_dev_time:.6f} seconds")

print("Embeddings updated successfully!")
