import os
import numpy as np
from config import client
from vector_store import get_chroma_collection, embedding_model
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Chargement du modèle de re ranking en local
reranker = CrossEncoder('BAAI/bge-reranker-base')

# Récupération ou Initialisation de la collection ChromaDB
collection = get_chroma_collection()

# Définition du seuil minimal pour le Guardrail 
SEUIL_PERTINENCE = 0.0

# Définition du System Prompt anti hallucination
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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def executer_pipeline_rag(input_data):
    # Gestion de l'historique de conversation
    if isinstance(input_data, str):
        conversation_history = [{"role": "user", "content": input_data}]
    else:
        conversation_history = input_data

    derniere_question = conversation_history[-1]["content"]

    # Réécriture de la requête pour spécifier une demande en rapport avec l'historique
    chat_model = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4o")
    
    rewrite_messages = [
        {
            "role": "system", 
            "content": (
                "Tu es un assistant expert en recherche technique. Ton rôle est de reformuler la dernière question "
                "de l'utilisateur en une requête de recherche textuelle autonome, explicite et technique, en te basant "
                "sur l'historique de la conversation si nécessaire. La requête doit intégrer les mots-clés essentiels "
                "(modèle d'iPhone, code d'erreur, capteur) pour être efficace dans une base documentaire de réparation. "
                "Réponds uniquement avec la requête reformulée, sans phrase de politesse ni mise en forme."
            )
        }
    ]
    rewrite_messages.extend(conversation_history)
    
    response_rewrite = client.chat.completions.create(
        model=chat_model,
        messages=rewrite_messages,
        temperature=0.0
    )
    query_recherchee = response_rewrite.choices[0].message.content.strip()
    print(f"\n[Query Rewriting] Question originale : {derniere_question}")
    print(f"[Query Rewriting] Requête optimisée pour la recherche : {query_recherchee}")

    # Vectorisation de la requête utilisateur
    question_vector = embedding_model.encode(query_recherchee).tolist()

    # Interrogation de ChromaDB récupération des 20 plus proches sémantiquement
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(20, collection.count())
    )
    retrieved_texts = results['documents'][0]

    # Préparation BM25 sur les documents restreints ramenés par ChromaDB
    tokenized_corpus = [doc.lower().split(" ") for doc in retrieved_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query_recherchee.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    normalized_bm25 = bm25_scores / max_bm25
    
    # Recherche hybride 
    scored_candidates = []
    alpha = 0.5 
    
    for i, text in enumerate(retrieved_texts):
        doc_vector = embedding_model.encode(text).tolist()
        cos_score = cosine_similarity(question_vector, doc_vector)
        kw_score = normalized_bm25[i]
        hybrid_score = (alpha * cos_score) + ((1 - alpha) * kw_score)
        
        scored_candidates.append({
            "text": text,
            "hybrid_score": hybrid_score
        })
        
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_candidates = scored_candidates[:10]
    
    # Cross Encoder basé sur la requête réécrite
    rerank_pairs = [[query_recherchee, cand["text"]] for cand in top_candidates]
    rerank_scores = reranker.predict(rerank_pairs)
    
    for idx, cand in enumerate(top_candidates):
        cand["rerank_score"] = float(rerank_scores[idx])
        
    top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    meilleur_candidat = top_candidates[0]
    score_max = meilleur_candidat["rerank_score"]
    
    print(f"[Pipeline RAG] Meilleur score de Re-ranking obtenu : {score_max:.4f}")
    
    # guardrail
    if score_max < SEUIL_PERTINENCE:
        print(f"[Guardrail Sécurité] Score ({score_max:.4f}) inférieur au seuil ({SEUIL_PERTINENCE}). Refus de répondre.")
        return (
            "**Avis de sécurité du RAG** : Je suis désolé, mais je n'ai pas trouvé d'information "
            "suffisamment pertinente ou vérifiée dans le guide de réparation pour répondre à cette question. "
            "Afin d'éviter toute erreur de diagnostic matériel, veuillez reformuler votre demande ou vérifier "
            "que le modèle d'appareil et le code d'erreur sont bien couverts par la base de connaissances."
        )
    
    best_match = meilleur_candidat["text"]
    print(f"[Guardrail Sécurité] Score validé. Transmission du contexte au LLM.")
    
    # Génération de la réponse
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "system", 
        "content": f"Contexte technique pertinent extrait et validé par le Re-ranker :\n{best_match}"
    })
    messages.extend(conversation_history)
    
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()