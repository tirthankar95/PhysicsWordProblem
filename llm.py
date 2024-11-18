import os
import json 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def load_env_vars():
    with open("config.json", "r") as file:
        env = json.load(file)
    for k, v in env.items():
        os.environ[k] = v

model_id = "microsoft/Phi-3.5-mini-instruct"
model_id = "HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceEndpoint(repo_id = model_id, temperature = 0.1)    

def StopHallucinations(response):
    return response.split("Question:")[0]

def get_question(problem):
    template = """
    {question}
    """
    prompt = PromptTemplate(intput = ["question"], template = template)
    rag_chain0 = (
        {"question": lambda x: x}
        | prompt
    )
    rag_chain1 = (
        {"question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
        | StopHallucinations
    )
    # print(rag_chain0.invoke(problem) )
    return rag_chain1.invoke(problem) 
