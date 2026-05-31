from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# Load a text embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample proprietary data
docs = [
    "Our business hours are from 9 AM to 5 PM.",
    "We provide logistics services for e-commerce.",
    "Customer support can be reached at support@example.com."
]

# Convert text to vector embeddings and index with FAISS
doc_vectors = np.array([embed_model.encode(doc) for doc in docs])
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(doc_vectors)

# Retrieve the top-k most relevant documents for the query
query = "What time does the company open?"
query_vector = np.array([embed_model.encode(query)])
D, I = index.search(query_vector, k=2)
retrieved_docs = [docs[i] for i in I[0]]
context = "\n".join(retrieved_docs)

# Generate a grounded answer using only the retrieved context (RAG)
response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": f"Answer using only the following context:\n{context}",
        },
        {"role": "user", "content": query},
    ],
)
print("Answer:", response["message"]["content"])
