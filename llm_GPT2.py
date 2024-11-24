from transformers import  AutoTokenizer, AutoModelForCausalLM 
from colorama import Style, Fore, Back 

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_question(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.1,
        max_length=256,
    )
    return tokenizer.batch_decode(gen_tokens)[0]