import os
from openai import OpenAI
from config import client

def generer_embeddings_azure(nodes):
    embedding_model = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    
    print(f"Vectorisation en cours avec le modèle Azure : {embedding_model}...")
    
    vectors_data = []
    for i, node in enumerate(nodes):
        response = client.embeddings.create(
            input=node.text,
            model=embedding_model
        )
        vector = response.data[0].embedding
        vectors_data.append({
            "id": i,
            "text": node.text,
            "vector": vector
        })
        print(f"Chunk n°{i + 1} vectorisé avec succès ({len(vector)} dimensions).")
        
    return vectors_data