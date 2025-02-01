import json 
import numpy as np
import pandas as pd 
import sys 
from collections import namedtuple
sys.path.insert(0, "LLM")
sys.path.insert(0, "LLM_CONFIG")
sys.path.insert(0, "UTILS")
from utils import GraphEquation, Env, fix
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
Topic = namedtuple("Topic", ["name", "env_name", "in_file", "out_file"])

def generate_question_variables(data):
    equations, units = data["equations"], {}
    obj = GraphEquation(equations)
    unknown, known, eqn = obj.getEquation()
    logging.debug(f"knwn: {known}, unknwn: {unknown}")
    problem = ""
    two_d = True if np.random.normal() >= 0 else False 
    for element in known:
        name, v_range, unit, type =  data['variable_names'][element]
        type = 0 if type == "S" else 1 # (0: scaler, 1: vector)
        var = fix(np.random.uniform(v_range[0], v_range[1]))
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

def get_solution(soln, data):
    soln = set(soln)
    solution = ""
    for idx, soln_id in enumerate(soln):
        if idx == 0: solution = data["equations"][soln_id]
        else: solution += ", " + data["equations"][soln_id]
    return solution

def get_phyQ(topic):
    load_env_vars()
    TRIALS = 25
    env_obj = Env(topic.env_name)
    with open(f"TOPICS/{topic.in_file}") as file:
        data = json.load(file)
    with open("LLM_CONFIG/config.json", "r") as file:
        env = json.load(file)
    try:
        df = pd.read_csv(f"DATASET/{topic.out_file}")
    except Exception as e:
        print(f'{Style.BRIGHT}{Fore.RED}Creating file...{Style.RESET_ALL}')
        with open(f"DATASET/{topic.out_file}", "w") as ofile:
            ofile.write("")
    for _ in range(TRIALS):
        prompt, soln, units_p = generate_question_variables(data)
        topic_words, units_t = env_obj.get_topic_words() 
        if np.random.normal() >= 0:
            topic_words = replace_words(topic_words, units_t)
        final_prompt = topic_words + " " + prompt  
        problem = get_question(final_prompt)
        logging.info(f'\n{Fore.MAGENTA}[PROMPT] {final_prompt.strip()}{Style.RESET_ALL}')
        print(f'{Style.BRIGHT}{Fore.CYAN}Is this question valid[y/n]?\n{Fore.GREEN}{problem.strip()}')
        print(f'{Fore.CYAN}[HINT] {get_solution(soln, data).strip()}{Style.RESET_ALL}')
        print('\n' + f'{Fore.BLACK}{Back.WHITE}--'*30 + f'{Style.RESET_ALL}' + '\n')
        if env["BUILD_DATASET"]:
            with open(f"DATASET/{topic.out_file}", "a") as file:
                new_row = {'Prompt':final_prompt, 'Question':problem}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
                print(f"{final_prompt}, {problem}", file = file)
        beautify(f"TOPICS/{topic.in_file}", data) 
    df.to_csv(f"DATASET/{topic.out_file}", index = False)

if __name__ == '__main__':
    topics = [
                Topic("SIMPLE KINEMATICS", "env_sm.json","simple_motion.json", "physics_sm.csv"), 
                Topic("NUCLEAR PHYSICS", "env_np.json","nuclear_physics.json", "physics_np.csv"), 
                Topic("GRAVITATION", "env_g.json","gravitation.json", "physics_g.csv")
            ]
    print(f'{Style.BRIGHT}{Fore.GREEN}Which topics do you want to generate questions from?\n')
    for i in range(len(topics)):
        print(f'{i}. {topics[i][0]}')
    print(f'{Style.RESET_ALL}')
    choice = int(input(f"{Style.BRIGHT}Choose the index of the topic: "))
    if choice < 0  or choice >= len(topics):
        print(f"{Style.BRIGHT}Wrong Choice!!\n")
    print(f'{Style.RESET_ALL}')
    get_phyQ(topics[choice])