# from utils.qa_chain import QAChain
import json, re
from langchain.prompts import PromptTemplate
from src.call_llm import get_completion

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
    请分析我用<>包围起来的问题，不需要回答问题本身，仅给出一个资料库供我参考，\
    特别注意：1. 以json格式返回dirname和alias，不要有任何其它的字符。
    2. 只需要给出一个资料库。
    3. 资料库仅限上所罗列，不要试图编造。
    <>
    {question}
    <>
"""
pattern = re.compile(r'\{[^\}]*\}')

prompt = PromptTemplate.from_template(default_prompt_template)

query = prompt.format(question = "请你给出雍州所辖的所有郡。")

response = get_completion(query, model='zhipu')
print (response)

search_result = re.findall(pattern, response)
print (search_result)

data = json.loads(search_result[0])
print (data)
print(type(data))

print(data["dirname"])