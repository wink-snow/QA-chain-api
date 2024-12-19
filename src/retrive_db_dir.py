import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json, re
from langchain.prompts import PromptTemplate
from src.call_llm import get_completion

PRE_MODEL = 'zhipu' # 默认的问题预处理模型

default_prompt_template = """
    我提供了如下用~~~包围起来的资料库：
    ~~~
            "dirname": "quhua", 
            "alias": "区划", 
            "description": "杞朝的行政区划，包括十八个州，二百五十八个郡或府"。
            "dirname": "guanzhi", 
            "alias": "官制", 
            "description": "杞朝的官制，包含中央官制、地方官制和诸王列侯制度"。
    ~~~
    请分析我用<>包围起来的问题，不需要回答问题本身，仅给出一个最可靠的资料库供我参考，\
    特别注意：1. 以json格式返回dirname和alias，不要有任何其它的字符。
    2. 只需要给出一个资料库。
    3. 资料库仅限上所罗列，不要试图编造。
    <>
    {question}
    <>
"""
pattern = re.compile(r'\{[^\}]*\}')

def get_retrived_dir(query: str):
    prompt = PromptTemplate.from_template(default_prompt_template)
    pre_query = prompt.format(question = query)

    response = get_completion(pre_query, model=PRE_MODEL)

    try:
        search_result = re.findall(pattern, response)
    except:
        raise Exception("预处理时json字符串匹配失败")
    
    try:
        data = json.loads(search_result[0])
        return data["dirname"]
    except:
        raise Exception("预处理时json字符串解析失败")
    
if __name__ == "__main__":
    query = "长安郡离京师多远？"
    print(get_retrived_dir(query))