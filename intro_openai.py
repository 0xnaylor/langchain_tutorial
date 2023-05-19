from dotenv import dotenv_values
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

# Load the environment variables
config = dotenv_values(".env")

os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
davinci = OpenAI(model_name='text-davinci-003')

# build prompt template for simple question-answering
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.run(question))