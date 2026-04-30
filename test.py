import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

#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
print("---------------------\n\tUSPESEN LOGIN\n---------------------")

# 1. Load the model (this will download once, then stay local)
# This model supports 50+ languages including Spanish, French, German, Chinese, etc.
model = SentenceTransformer("google/embeddinggemma-300m")

# 2. Your vocabulary database (can be mixed languages)
vocab_data = [
    {"word": "apple", "lang": "en"},
    {"word": "manzana", "lang": "es"},
    {"word": "perro", "lang": "es"},
    {"word": "dog", "lang": "en"},
    {"word": "Katze", "lang": "de"},
    {"word": "bicycle", "lang": "en"},
    {"word": "Fahrrad", "lang": "de"},
    {"word": "running", "lang": "en"}
]

# Extract just the words for embedding
words = [item['word'] for item in vocab_data]

# 3. Create Embeddings
print("Encoding vocabulary...")
embeddings = model.encode(words)

# 4. Set up the FAISS Index (Local Vector DB)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
index.add(np.array(embeddings).astype('float32'))

def search_vocabulary(query, top_k=3):
    # Embed the query
    query_vector = model.encode([query])
    
    # Search the index
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    
    print(f"\nResults for search: '{query}'")
    for i in range(top_k):
        idx = indices[0][i]
        word_info = vocab_data[idx]
        print(f" - {word_info['word']} ({word_info['lang']}) | Score: {distances[0][i]:.4f}")

# --- Test the prototype ---
search_vocabulary("a pet that barks") # Should find dog/perro
search_vocabulary("two-wheeled transport") # Should find bicycle/Fahrrad