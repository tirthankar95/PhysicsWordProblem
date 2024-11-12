import json 
import numpy as np 
import nltk
import os
from nltk.corpus import wordnet 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
operators = ['+', '*', '=']

def load_env_vars():
    with open("config.json", "r") as file:
        env = json.load(file)
    for k, v in env.items():
        os.environ[k] = v

def generate_question(filename):
    with open(f"{filename}") as file:
        data = json.load(file)
    euqations = data["equations"]
    equation = euqations[np.random.randint(0, len(euqations))]
    elements_, elements = [], [var for var in equation.split(' ') if var not in operators]
    for x in elements:
        try:
            var = float(x)
        except:
            elements_.append(x)
    unknown_element = elements_[np.random.randint(0, len(elements_))]
    elements = [var for var in elements_ if var not in unknown_element]
    problem = "Generate a physics question using the following elements.\n"
    for element in elements:
        name, v_range, unit =  data['variable_names'][element]
        var = np.random.randint(v_range[0], v_range[1])
        problem += f"{name} = {var} {unit}, "
    name, v_range, unit = data['variable_names'][unknown_element]
    problem += f"{name} = ?"
    return problem, data

######################################## LLM ####################################################

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

######################################## LLM ####################################################


def update_data(filename, data):
    with open(f"{filename}", "w") as file:
        json.dump(data, file, indent=5)

if __name__ == '__main__':
    problem, data = generate_question("simple_motion.json")
    TRIALS = 1
    for _ in range(TRIALS):
        problem = get_question(problem)
        print(f'Is this question valid[y/n]:\n{problem}')
        ch = input()
        if ch == 'y':
            data['question_bank'].append(problem)
        update_data("simple_motion.json", data)