import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.get_vectordb import load_vector_db
from src.models_to_llm import get_llm

class SelfQAChainForChat():
    """
    拥有历史对话记录的检索问答链。

    接收参数：
        `model_name`: 模型名称，可选参数：`spark`,`zhipu`.
        `persit_path`: 向量数据库的保存路径。
        `top_k`: 检索时返回的结果数量。
        `history_len`: 保留的历史对话记录长度。
    """

    default_question_condense_template = """为你提供了如下对话记录和一个问题，请改述使之成为一个独立的问题：
        历史对话记录:
        {chat_history}
        当前问题: {question}
        改述后的问题：
    """

    def __init__(self, model_name: str, persit_path: str, top_k: int = 2, history_len: int = 5):
        self.llm = get_llm(model=model_name)
        self.vectordb = load_vector_db(persit_path)
        self.top_k = top_k

        self.chat_history = []
        self.history_len = history_len

        self.retriever = self.vectordb.as_retriever(
            search_type = "similarity",
            search_kwargs = {
                "k": self.top_k
            }
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=PromptTemplate.from_template(self.default_question_condense_template),
            memory=self.memory
        )


    def cut_history(self):
        """
        保留一定长度的历史对话记录。
        """

        if len(self.chat_history) <= self.history_len:
            return self.chat_history
        n = len(self.chat_history)
        return self.chat_history[n-self.history_len-1:]
    
    def answer(self, question: str = None):
        if len(question) == 0:
            return ""
        
        self.chat_history = self.cut_history()

        result = self.qa_chain(
            {
                "question": question
            }
        )
        answer = result["answer"]
        self.chat_history.append((question, answer))
        return answer