import gradio as gr
from rag_pipeline import executer_pipeline_rag

def predire_diagnostic(message, history):
    messages_formates = []
    
    # Reconstitution de l'historique au format attendu par le RAG
    for user_msg, assistant_msg in history:
        messages_formates.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages_formates.append({"role": "assistant", "content": assistant_msg})
            
    messages_formates.append({"role": "user", "content": message})
    
    # Appel du pipeline RAG
    reponse = executer_pipeline_rag(messages_formates)
    return reponse

# Configuration de l'interface Gradio standard et universelle
demo = gr.ChatInterface(
    fn=predire_diagnostic,
    title="Assistant RAG - Diagnostic iPhone Panic-Full",
    description="Système d'analyse sémantique et lexicale de rapports de crash iOS.",
    examples=[
        "Mon iPhone 14 Pro a un kernel panic avec le code 0x80000, c'est quel composant ?",
        "Quelles sont les causes d'un redémarrage en boucle sur un iPhone 13 ?"
    ]
)

if __name__ == "__main__":
    print("=== Lancement de l'interface graphique Gradio ===")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)