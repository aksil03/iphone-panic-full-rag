import os
import numpy as np
from config import client
from ingestion import charger_et_decouper_documents
from vector_store import generer_embeddings_azure
from rank_bm25 import BM25Okapi

# Définition du System Prompt anti-hallucination
SYSTEM_PROMPT = """
Tu es un assistant expert en diagnostic et réparation de cartes mères et de matériel iPhone. 
Tu analyses des rapports de crash (panic-full) fournis par les utilisateurs.

Instructions de diagnostic :
1. **Priorité aux sources** : Appuie-toi en priorité absolue sur le contexte technique fourni (correspondances de capteurs, codes hexadécimaux, bus I2C) pour identifier la panne matérielle précise.
2. **Complémentarité** : Si un terme technique ou une erreur générale d'iOS apparaît dans le log mais n'est pas explicitement détaillé dans le contexte, tu peux utiliser tes connaissances générales pour expliquer ce qu'il signifie, à condition de préciser qu'il s'agit d'une analyse générale.
3. **Sécurité / Anti-hallucination matérielle** : N'invente **jamais** de correspondance de code hexadécimal (ex: 0x800, 0x40000) ni de diagnostic de composant spécifique s'il n'est pas validé par le contexte. En cas de doute, reste prudent et suggère des vérifications méthodiques.
4. **Clarté** : Structure ton diagnostic clairement : 
   - Cause probable (selon la base de connaissances ou l'analyse)
   - Composants ou lignes à vérifier
   - Pistes de résolution en atelier
"""

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def executer_pipeline_rag(input_data):
    # Si l'utilisateur passe une simple string, on la convertit en format historique
    if isinstance(input_data, str):
        conversation_history = [{"role": "user", "content": input_data}]
    else:
        conversation_history = input_data

    # On extrait la dernière question posée pour effectuer la recherche RAG pertinente
    derniere_question = conversation_history[-1]["content"]

    # Chargement et découpage des documents
    nodes = charger_et_decouper_documents()
    
    # Préparation du corpus pour BM25 recherche par mots-clés exacts
    corpus_textes = [node["text"] for node in nodes]
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_textes]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Génération des embeddings des chunks
    vectors_data = generer_embeddings_azure(nodes)
    
    # Vectorisation de la dernière question de l'utilisateur
    embedding_model = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    response_q = client.embeddings.create(input=derniere_question, model=embedding_model)
    question_vector = response_q.data[0].embedding
    
    # Calcul des scores BM25 sur la dernière question
    tokenized_query = derniere_question.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    normalized_bm25 = bm25_scores / max_bm25
    
    # Recherche Hybride Combinaison Cosinus + BM25
    best_match = None
    highest_hybrid_score = -1
    alpha = 0.5  # Équilibre (50% sémantique, 50% mots-clés exacts)
    
    for i, item in enumerate(vectors_data):
        cos_score = cosine_similarity(question_vector, item["vector"])
        kw_score = normalized_bm25[i]
        
        # Score hybride combiné
        hybrid_score = (alpha * cos_score) + ((1 - alpha) * kw_score)
        
        if hybrid_score > highest_hybrid_score:
            highest_hybrid_score = hybrid_score
            best_match = item["text"]
            
    print(f"\n[RAG Hybride avec Mémoire] Meilleur score combiné : {highest_hybrid_score:.2f}")
    
    # Construction de la charge utile de messages pour GPT-4o (System Prompt + Contexte RAG + Historique complet)
    chat_model = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4o")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # On injecte le contexte technique récupéré sous forme de message système d'appoint
    messages.append({
        "role": "system", 
        "content": f"Contexte technique pertinent extrait de la base de connaissances pour t'aider dans la conversation :\n{best_match}"
    })
    
    # On ajoute toute la conversation précédente pour que le modèle garde le fil
    messages.extend(conversation_history)
    
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.0  # sans fantaisie
    )
    
    return response.choices[0].message.content.strip()