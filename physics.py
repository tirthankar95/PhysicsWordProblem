import json 
import numpy as np
import pandas as pd 
from utils import GraphEquation, Env
from llm import get_question
from colorama import Fore, Back, Style 
from word_change import replace_words
from utils import load_env_vars
import logging
logging.basicConfig(
     level=logging.INFO, 
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

def generate_question_variables(data):
    equations, units = data["equations"], {}
    obj = GraphEquation(equations)
    unknown, known, eqn = obj.getEquation()
    problem = ""
    two_d = True if np.random.normal() >= 0 else False 
    for element in known:
        name, v_range, unit, type =  data['variable_names'][element]
        type = 0 if type == "S" else 1 # (0: scaler, 1: vector)
        var = np.random.randint(v_range[0], v_range[1])
        if two_d and type == 1: 
            theta = np.random.randint(0, 180)
            problem += f"{name} = {var} {unit} at an angle {theta} degrees with the horizontal, "
        else:
            problem += f"{name} = {var} {unit}, "
        units[unit] = True
    for idx, element in enumerate(unknown):
        name, v_range, unit, type = data['variable_names'][element]
        type = 0 if type == "S" else 1 # (0: scaler, 1: vector)
        if two_d and type == 1: 
            problem += f"horizontal component of {name} = unknown, " +\
                        f"vertical component of {name} = unknown, "
        else:
            problem += f"{name} = unknown, "
    problem = problem[:-2] + "."
    logging.debug(problem)
    return problem, eqn, units

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
    load_env_vars()
    TRIALS = 1
    env_obj = Env()
    with open("dataset/simple_motion.json") as file:
        data = json.load(file)
    with open("config.json", "r") as file:
        env = json.load(file)
    df = pd.read_csv("dataset/physicsQ.csv")
    for _ in range(TRIALS):
        prompt, soln, units_p = generate_question_variables(data)
        topic_words, units_t = env_obj.get_topic_words() 
        if np.random.normal() >= 0:
            topic_words = replace_words(topic_words, units_t)
        final_prompt = topic_words + " " + prompt  
        problem = get_question(final_prompt)
        print(f'{Fore.MAGENTA}[PROMPT] {final_prompt}{Style.RESET_ALL}')
        print(f'{Fore.CYAN}[HINT] {get_solution(soln)}{Style.RESET_ALL}')
        print(f'{Style.BRIGHT}{Fore.CYAN}Is this question valid[y/n]?\n{Fore.GREEN}{problem}')
        print('\n' + f'{Fore.BLACK}{Back.WHITE}--'*30 + f'{Style.RESET_ALL}' + '\n')
        ch = 'y' #input() if env["BUILD_DATASET"] else 'n'  
        if ch == 'y':
            with open("dataset/physicsQ.csv", "a") as file:
                new_row = {'Prompt':final_prompt, 'Question':problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
                print(f"{final_prompt}, {problem}", file = file)
        beautify("dataset/simple_motion.json", data) 
    print(df.head().T)
    df.to_csv("dataset/physicsQ.csv", index = False)
