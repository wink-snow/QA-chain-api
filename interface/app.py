import streamlit as st
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from src.call_llm import get_completion
from src.retrive_db_dir import get_retrived_dir
from utils.qa_chain import QAChain
from utils.inter_qa_chain_for_chat import InterQAChainForChat

PERSIST_ROOT_PATH = './db/vector_db/chroma/'
PERSIST_PATH = './db/vector_db/chroma/quhua/'

def get_response(input_text, model_name='zhipu'):
    response = get_completion(prompt=input_text, model=model_name)
    return response

def generate_response_qa_chain(input_text):
    retrived_dirname = get_retrived_dir(input_text)
    qa_chain = QAChain(model_name='zhipu', vectordb_path=PERSIST_ROOT_PATH + retrived_dirname + '/', top_k=8)
    response = qa_chain.answer(input_text)
    return response

def main():
    st.set_page_config(
        page_title="雪墨杞言",
        page_icon="❄"
    )

    st.title('💭杞小助的问答界面')

    selected_method = st.sidebar.selectbox(
        "你想选择哪种模式进行对话？",
        ["普通AI", "无记忆的杞小助（推荐）", "有记忆的杞小助（开发中）"],
        index=1
    )

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    if 'chain_for_chat' not in st.session_state:
        st.session_state['chain_for_chat'] = InterQAChainForChat(model_name='zhipu', top_k=8)

    messages = st.container(height=300)
    chat_input = st.chat_input("说点什么吧，杞朝的小伙伴")
    footer = """
            <style>
                .st-footer {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 50px;
                    background-color: white;
                }
                .st-footer-content {
                    font-size: 12px;
                }
            </style>
            <div class="st-footer">
                <div class="st-footer-content">
                    <p><a href="https://beian.miit.gov.cn/" target="_blank">陕ICP备2024041464号-1</a></p>
                </div>
            </div>
            """

    st._bottom.markdown(footer, unsafe_allow_html=True)
    if prompt := chat_input:
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == '普通AI':
            answer = get_response(prompt)
        elif selected_method == '无记忆的杞小助（推荐）':
            answer = generate_response_qa_chain(prompt)
        elif selected_method == '有记忆的杞小助（开发中）':
            answer = st.session_state.chain_for_chat.answer(prompt)
        else:
            answer = None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])

if __name__ == '__main__':
    main()