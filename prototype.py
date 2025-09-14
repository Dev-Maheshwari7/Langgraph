# floatchat.py
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
#from langchain_community.llms import Ollama
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
# 1. LLM + Embeddings
llm = OllamaLLM(model="llama2")
embeddings = OllamaEmbeddings(
    model="llama2",
)

# 2. Load docs (after NetCDF → metadata text)
loader = TextLoader("argo_metadata.txt")
docs = loader.load()

# 3. Vector DB
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# 4. Retrieval + QA chain
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 5. LangGraph state
class State(dict): pass

graph = StateGraph(State)

def retrieve_node(state: State):
    query = state["question"]
    result = qa.run(query)
    state["retrieval_result"] = result
    return state

def sql_node(state: State):
    query = state["question"]
    sql_prompt = f"Write a PostgreSQL query for ARGO float data: {query}"
    state["sql"] = llm.invoke(sql_prompt).content
    return state

graph.add_node("retrieve", retrieve_node)
graph.add_node("sql", sql_node)

graph.add_edge("retrieve", "sql")
graph.set_entry_point("retrieve")
graph.set_finish_point("sql")

app = graph.compile()

# Demo
if __name__ == "__main__":
    question = "Show me salinity profiles near the equator in March 2023"
    result = app.invoke( "Show me salinity profiles near the equator in March 2023")
    print("Answer:", result["retrieval_result"])
    print("SQL:", result["sql"])
