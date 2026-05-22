from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Write me the summary of the poem \n {poem}',
    input_variables = ['poem']

)

loader = TextLoader('cricket.txt',encoding = 'utf8')

docs = loader.load()

chain = prompt | model | parser 

result = chain.invoke({'poem':docs[0].page_content})

print(result)

# print(docs)

# print(type(docs))

# print(len(docs))

# print(type(docs[0]))