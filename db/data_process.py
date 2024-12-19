"""
    Split PDF files into chunks and clean the text data.
"""
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os, re

DEFAULT_CHUNK_SIZE = 200 # 单个文本块大小
OVERLAP_SIZE = 60 # 文本块重叠大小

def get_file_paths(folder_path: str):
    """
    获取`folder_path`路径下所有文件的路径。
    """

    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def clean_data(single_doc: Document = None):
    """
    清洗文本数据。
    """

    default_pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    single_doc.page_content = re.sub(default_pattern, lambda match: match.group(0).replace('\n', ''), single_doc.page_content)
    return single_doc

def data_process(folder_path: str):
    """
    加载`folder_path`路径下所有文件，并使用`text_splitter`进行文本分割。

    请求参数：
        folder_path (str): 文件夹路径
    """
    
    file_paths = get_file_paths(folder_path)

    loaders = []
    for file_path in file_paths:
        filetype = file_path.split(".")[-1]
        if filetype == "pdf":
            loaders.append(PyMuPDFLoader(file_path))
        else:
            return ValueError(f"Unsupported file type: {filetype}")
        
    texts = []
    for loader in loaders:
        text = []
        for page in loader.load():
            page_content = clean_data(page)
            text.append(page_content)
        texts.extend(text)

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = DEFAULT_CHUNK_SIZE,
        chunk_overlap  = OVERLAP_SIZE
    )
    split_docs = text_splitter.split_documents(texts)
    return split_docs

if __name__ == "__main__":
    file_folder = './assets/new/'
    split_docs = data_process(folder_path=file_folder)
    print(f"分割成{len(split_docs)}个文本块")
    for doc in split_docs:
        print(doc.page_content)
        print("\n——————————————\n")