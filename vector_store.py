import chromadb
from sentence_transformers import SentenceTransformer
from ingestion import charger_et_decouper_documents

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="iphone_repair_docs")

# Chargement du modèle d'embedding léger
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_chroma_collection():
    # Initialise et indexe les documents dans ChromaDB s'ils n'y sont pas déjà puis retourne la collection
    if collection.count() == 0:
        print("[ChromaDB] Base vide. Indexation des documents en cours...")
        nodes = charger_et_decouper_documents()
        
        ids, texts, embeddings = [], [], []
        for i, node in enumerate(nodes):
            vector = embedding_model.encode(node.text).tolist()
            ids.append(f"doc_{i}")
            texts.append(node.text)
            embeddings.append(vector)
            
        collection.add(ids=ids, documents=texts, embeddings=embeddings)
        print(f"[ChromaDB] {len(nodes)} chunks indexés avec succès dans la base !")
    else:
        print(f"[ChromaDB] Base déjà chargée ({collection.count()} chunks trouvés dans la base persistante).")
        
    return collection