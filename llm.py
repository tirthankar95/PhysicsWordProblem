import json 
from colorama import Style, Fore
import llm_GPT2, llm_RAG_Hgf, llm_OpenAI

with open("config.json", "r") as file:
    data = json.load(file)

def get_question(question):
    if data["LLM_MODEL"]["MAIN_MODEL"] == "llm_GPT2":
        return llm_GPT2.get_question(question)
    if data["LLM_MODEL"]["MAIN_MODEL"] == "llm_RAG":
        return llm_RAG_Hgf.get_question(question, data["LLM_MODEL"]["INNER_MODEL"])
    if data["LLM_MODEL"]["MAIN_MODEL"] == "llm_OpenAI":
        return llm_OpenAI.get_question(question, data["LLM_MODEL"]["INNER_MODEL"])