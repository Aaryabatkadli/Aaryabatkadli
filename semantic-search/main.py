from fastapi import FastAPI
import requests
from vector_store import VectorStore


app = FastAPI()


OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "theepicdev/nomic-embed-text:v1.5-q6_K"


vector_db = None



def get_embedding(text: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()  # helps catch errors early
    return response.json()["embedding"]



@app.on_event("startup")
def startup_event():
    global vector_db

    # Sample paragraph to store
    paragraph = "GlideCloud promotes a culture of continuous learning, collaboration, and innovation. Employees are encouraged to experiment with new technologies and improve their technical skills regularly."

    # Get embedding
    embedding = get_embedding(paragraph)

    # Create vector database
    vector_db = VectorStore(dimension=len(embedding))

    # Store paragraph
    vector_db.add(embedding, paragraph)

    print("✅ Vector database initialized")



@app.post("/search")
def search(query: str):
    query_embedding = get_embedding(query)
    result = vector_db.search(query_embedding)
    return {
        "query": query,
        "result": result
    }



@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/debug/db")
def view_database():
    return {
        "total_vectors": vector_db.index.ntotal,
        "stored_texts": vector_db.texts
    }
