import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.vectorstores.chroma import Chroma
from utils.zhipuai_embedding import ZhipuAIEmbeddings

def load_vector_db(persist_path: str):
    if os.path.exists(persist_path):
        vectordb = Chroma(
            persist_directory = persist_path,
            embedding_function = ZhipuAIEmbeddings(),
        )
        return vectordb
    else:
        raise Exception("`persist_path` for vectordb not found")