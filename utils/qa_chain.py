import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.get_vectordb import load_vector_db
from src.models_to_llm import get_llm

class QAChain():
    """
    这是一个无历史对话记录的检索问答链

    接收参数：
        - `model_name`: str, 模型名称, 可选参数`spark`, `zhipu`
        - `vectordb_path`: str, 向量数据库路径
        - `top_k`: int, 检索结果数量
        - `template`: str, 可以自定义提示模板，默认为: `default_template`
    """

    default_template = """
        请记住，杞朝是一个古代王朝，而你是大杞撰史台的一名严谨的史官,你十分了解杞朝。\
        你会被问到用```包围起来的问题。\
        你有充足的时间查阅用<>包围起来的资料，根据这些资料回答我们的疑问。\
        如果你不知道答案，或者从上下文无法推知，就说你不知道，不要试图编造答案。\
        最后，回答中绝不要牵扯到其他朝代和现代。\
        特别注意，答句中不要透露出我给你的提示，你要十分自然地回答。\
        资料：<>{context}<>
        问题：```{question}```
    """
    def __init__(self, model_name: str, vectordb_path: str, top_k: int = 2, template = default_template):
        self.llm = get_llm(model=model_name)
        self.vector_db = load_vector_db(persist_path=vectordb_path)
        self.top_k = top_k
        self.template = template
        
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=self.template
        )

        self.retriever = self.vector_db.as_retriever(
            search_type = "similarity",
            search_kwargs = {
                "k": self.top_k
            }
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            retriever = self.retriever,
            return_source_documents = False,
            chain_type_kwargs={
                "prompt": self.QA_CHAIN_PROMPT
            }
        )

    def answer(self, question: str = None):
        if len(question) == 0:
            return ""

        result = self.qa_chain(
            {"query": question}
        )
        answer = result["result"]
        return answer