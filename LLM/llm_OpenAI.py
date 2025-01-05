from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import load_env_vars

def get_question(problem, m_name):
    llm = OpenAI(model_name = f"{m_name}") 
    template = """
    {question}
    """
    prompt = PromptTemplate(intput = ["question"], template = template)
    rag_chain = (
        {"question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(problem) 

if __name__ == '__main__':
    load_env_vars()
    print(get_question("Who are you?", "gpt-3.5-turbo-instruct"))