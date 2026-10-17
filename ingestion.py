from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

def charger_et_decouper_documents():
    # Charge le guide depuis le dossier data
    documents = SimpleDirectoryReader("data").load_data()
    
    # Découpage par sections Markdown
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"{len(nodes)} chunks créés à partir du guide de réparation.")
    return nodes