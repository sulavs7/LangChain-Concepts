from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage ,HumanMessage, AIMessage

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report 
template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template = "write a 5 line summary on following text. /n{text}",
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic':'black_hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)

