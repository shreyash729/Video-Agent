import sys
import os

sys.path = [p for p in sys.path if '/agents/' not in p]
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('opentelemetry'):
        del sys.modules[mod_name]

# Compatibility imports: some langchain ecosystem packages have moved/renamed.
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
from core.config import get_embedding_instance

def get_embeddings(config: dict):
    return get_embedding_instance(config)



def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )

