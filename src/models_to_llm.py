from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.call_llm import gen_spark_params
from langchain_community.llms.sparkllm import SparkLLM
from utils.zhipuai_llm import ZhipuAILLM

def get_llm(model: str = None):

    if model == 'spark':
        llm = SparkLLM(spark_api_url=gen_spark_params(model='spark-v3.5')["spark_url"],
                    spark_app_id=os.environ["SPARK_APPID"],
                    spark_api_key=os.environ["SPARK_API_KEY"],
                    spark_api_secret=os.environ["SPARK_API_SECRET"],
                    top_k=1
                    )
        
    elif model == 'zhipu':
        zhipu_api_key = os.environ["ZHIPUAI_API_KEY"]
        llm = ZhipuAILLM(model="glm-4-air", api_key=zhipu_api_key)

    else:
        raise ValueError(f"Unsupported model: {model}")
    
    return llm