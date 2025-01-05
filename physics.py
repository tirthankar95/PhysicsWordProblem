import json 
import numpy as np
import pandas as pd 
import sys 
sys.path.insert(0, "LLM")
sys.path.insert(0, "LLM_CONFIG")
sys.path.insert(0, "UTILS")
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
    with open("TOPICS/simple_motion.json") as file:
        data = json.load(file)
    with open("LLM_CONFIG/config.json", "r") as file:
        env = json.load(file)
    df = pd.read_csv("DATASET/physicsQ.csv")
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
        if env["BUILD_DATASET"]:
            with open("DATASET/physicsQ.csv", "a") as file:
                new_row = {'Prompt':final_prompt, 'Question':problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
                print(f"{final_prompt}, {problem}", file = file)
        beautify("TOPICS/simple_motion.json", data) 
    df.to_csv("DATASET/physicsQ.csv", index = False)

#         final_prompt = """Give me a physics question using the following variables and words.

#  In a cave, person, ball, rock, cave length = 1271 m. displacement = 47 m, time = 62 s, final velocity = 53 m/s, force = 80 N, initial velocity = unknown, acceleration = unknown, mass = unknown.
# """