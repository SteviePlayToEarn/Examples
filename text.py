import openai
import numpy as np
from numpy.linalg import norm

openai.api_key = "YOUR_OPENAI_API_KEY"

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# === Beispieltexte ===
texts = [
    "The cat sits on the mat.",
    "A dog is playing in the garden.",
    "Artificial intelligence is transforming technology.",
    "This is a simple sentence about a cat.",
    "Quantum computing is the future of computation."
]

# === Vektoren für Beispieltexte erzeugen ===
embeddings = [get_embedding(text) for text in texts]

# === Suchtext ===
query = "Cats are relaxing on carpets."
query_embedding = get_embedding(query)

# === Ähnlichkeiten berechnen ===
similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]

# === Ergebnisse sortieren und ausgeben ===
sorted_results = sorted(zip(texts, similarities), key=lambda x: x[1], reverse=True)

print("Top similar texts:")
for text, score in sorted_results[:3]:
    print(f"- {text} (similarity: {score:.3f})")
