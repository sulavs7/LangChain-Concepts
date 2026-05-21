from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables = {'topic'}
)

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)

prompt2 = PromptTemplate(
    template = "explain me this joke \n{text}",
    input_variables = ['text']
)

parser = StrOutputParser()
chain = RunnableSequence(prompt , model , parser, prompt2, model , parser)
print(chain.invoke({'topic':'AI'}))