from sentence_transformers import SentenceTransformer

# Chargement d'un modèle d'embedding léger
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def generer_embeddings_azure(nodes):
    print("Vectorisation en cours en local sur le CPU (SentenceTransformers)...")
    
    vectors_data = []
    for i, node in enumerate(nodes):
        # Génération du vecteur en local, conversion du tableau numpy en liste Python
        vector = embedding_model.encode(node.text).tolist()
        
        vectors_data.append({
            "id": i,
            "text": node.text,
            "vector": vector
        })
        print(f"Chunk n°{i + 1} vectorisé en local ({len(vector)} dimensions).")
        
    return vectors_data