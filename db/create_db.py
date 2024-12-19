import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.zhipuai_embedding import ZhipuAIEmbeddings
from utils.handsome_data_processor import HandsomeDataProcessor
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

DATA_FOLDER_PATH = './assets/new/'
PERSIST_DIRECTORY = './db/vector_db/chroma/guanzhi/'

def get_split_docs():
    processor = HandsomeDataProcessor(DATA_FOLDER_PATH)
    return processor.data_process()

def create_db(embedding = ZhipuAIEmbeddings()):
    """
    创建向量数据库。

    返回：
        vectordb (Chroma): 向量数据库实例。
    """

    split_docs = get_split_docs()

    vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY  
    )

    vectordb.persist()
    return vectordb

if __name__ == '__main__':
    vectordb = create_db()