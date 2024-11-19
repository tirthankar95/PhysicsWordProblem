import json 
import numpy as np
import pandas as pd 
from utils import GraphEquation
from llm import get_question
from nltk.corpus import wordnet 
from colorama import Fore, Back, Style 
import logging
logging.basicConfig(
     level=logging.INFO, 
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

def generate_question_variables(data):
    equations = data["equations"]
    obj = GraphEquation(equations)
    unknown, known, eqn = obj.getEquation()
    problem = "Generate one physics question using the following elements.\n"
    for element in known:
        name, v_range, unit =  data['variable_names'][element]
        var = np.random.randint(v_range[0], v_range[1])
        problem += f"{name} = {var} {unit}, "
    for idx, element in enumerate(unknown):
        name, v_range, unit = data['variable_names'][element]
        if idx < len(unknown)-1: problem += f"{name} = unknown, "
        else: problem += f"{name} = unknown."
    logging.debug(problem)
    return problem, eqn

def beautify(filename, data):
    with open(f"{filename}", "w") as file:
        json.dump(data, file, indent=5)

def get_solution(soln):
    soln = set(soln)
    solution = ""
    for idx, soln_id in enumerate(soln):
        if idx == 0: solution = data["equations"][soln_id]
        else: solution += ", " + data["equations"][soln_id]
    return solution

if __name__ == '__main__':
    TRIALS = 5
    with open("dataset/simple_motion.json") as file:
        data = json.load(file)
    with open("config.json", "r") as file:
        env = json.load(file)
    df = pd.read_csv("dataset/physicsQ.csv")
    for _ in range(TRIALS):
        prompt, soln = generate_question_variables(data)
        problem = prompt # get_question(prompt)
        print(f'{Fore.MAGENTA}[PROMPT] {prompt}{Style.RESET_ALL}')
        print(f'{Fore.CYAN}[Soln] {get_solution(soln)}{Style.RESET_ALL}')
        print(f'{Style.BRIGHT}{Fore.CYAN}Is this question valid[y/n]?\n{Fore.GREEN}{problem}')
        print('\n' + f'{Fore.BLACK}{Back.WHITE}--'*30 + f'{Style.RESET_ALL}' + '\n')
        ch = input() if env["BUILD_DATASET"] else 'n'  
        if ch == 'y':
            with open("dataset/physicsQ.csv", "a") as file:
                new_row = {'Prompt':prompt, 'Question':problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
                print(f"{prompt}, {problem}", file = file)
        beautify("dataset/simple_motion.json", data) 
    print(df.head().T)
    df.to_csv("dataset/physicsQ.csv", index = False)
