from rag_pipeline import executer_pipeline_rag

if __name__ == "__main__":
    print("=== Démarrage de l'assistant RAG iPhone ===")
    
    # Exemple de question sur un code de crash
    question_test = "Mon iPhone 14 Pro a un kernel panic avec le code 0x80000, c'est quel composant ?"
    
    print(f"\nQuestion posée : {question_test}")
    
    reponse = executer_pipeline_rag(question_test)
    
    print("\n--- Diagnostic de l'IA ---")
    print(reponse)