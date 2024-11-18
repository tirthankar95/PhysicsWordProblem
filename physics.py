import json 
import nltk
import numpy as np
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

def update_data(filename, data):
    with open(f"{filename}", "w") as file:
        json.dump(data, file, indent=5)

if __name__ == '__main__':
    TRIALS = 5
    with open("simple_motion.json") as file:
        data = json.load(file)
    for _ in range(TRIALS):
        problem, soln = generate_question_variables(data)
        problem = get_question(problem)
        print(f'{Style.BRIGHT}{Fore.CYAN}Is this question valid[y/n]?\n{Fore.GREEN}{problem}')
        print('\n' + f'{Fore.BLACK}{Back.WHITE}--'*30 + f'{Style.RESET_ALL}' + '\n')
        ch = 'n' # input()
        if ch == 'y':
            data['question_bank'].append(problem)
        update_data("simple_motion.json", data)