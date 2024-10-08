import chromadb
import time
import statistics

txt_file_path = "chun0.txt"

with open(txt_file_path, "r", encoding="utf-8") as file:
    texts = [line.strip() for line in file.readlines()]

client = chromadb.PersistentClient(path="/c/Users/Sergi Jaume/Desktop/CBDE/CBDEPr-ctica1")

collection = client.get_or_create_collection(name="text_data_collection")

times = []

for i, text in enumerate(texts):
    start_time = time.time()
    print(i)
    collection.add(
        ids=[f"id_{i}"],
        documents=[text],
        embeddings=None,
        metadatas=[{"text": text}]
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    times.append(elapsed_time)

min_time = min(times)
max_time = max(times)
avg_time = statistics.mean(times)
std_dev_time = statistics.stdev(times)

print(f"Minimum time to store a document: {min_time:.6f} seconds")
print(f"Maximum time to store a document: {max_time:.6f} seconds")
print(f"Average time to store a document: {avg_time:.6f} seconds")
print(f"Standard deviation of times: {std_dev_time:.6f} seconds")

print ("Data loaded!")