# iPhone Diagnostic & Repair RAG Assistant (Panic-Full Logs)

## Aperçu du Projet
Ce projet est un assistant intelligent basé sur l'IA générative et l'architecture **RAG (Retrieval-Augmented Generation)**. Il est conçu pour aider les techniciens et réparateurs à analyser les rapports de crash (erreurs *Panic-Full* et codes hexadécimaux) des cartes mères iPhone afin d'identifier précisément les composants défectueux.

---

## Architecture & Fonctionnalités Techniques
* **Recherche Hybride Avancée** : Combine la recherche sémantique par embeddings vectoriels (Azure OpenAI / `text-embedding-3-small`) et la recherche par mots-clés exacts (`BM25Okapi`) pour une précision maximale sur les codes d'erreur techniques.
* **Orchestration & Découpage** : Utilisation de **LlamaIndex** pour ingérer et structurer la base de connaissances de réparation.
* **Modèle de Langage (LLM)** : Exploitation de **GPT-4o** (via Azure OpenAI) avec un System Prompt strict anti-hallucination et une température à `0.0` pour garantir la fiabilité des diagnostics matériels.
* **Gestion de l'Historique** : Support de la mémoire conversationnelle pour affiner les diagnostics au fil des échanges.

---

## Structure du Dépôt
* `config.py` : Configuration de la connexion au client Azure OpenAI.
* `ingestion.py` : Chargement et découpage des documents de la base de connaissances.
* `vector_store.py` : Génération des vecteurs d'embeddings.
* `rag_pipeline.py` : Logique de recherche hybride et orchestration du pipeline RAG avec GPT-4o.
* `main.py` : Script d'exécution principal pour lancer les tests de diagnostic.
* `data/` : Dossier contenant la base de connaissances technique.

---

## Guide d'installation et d'utilisation

Cloner le dépôt :
   ```bash
   git clone [https://github.com/aksil03/iphone-panic-full-rag.git](https://github.com/aksil03/iphone-panic-full-rag.git)
