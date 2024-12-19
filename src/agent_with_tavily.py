import os, sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models_to_llm import get_llm

LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

search = TavilySearchResults(max_results=2)
tools = [search]
llm = get_llm(model='zhipu')