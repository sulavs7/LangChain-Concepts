from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = ' Answer the following question\n{question} from the given text \n{text}',
    input_variables = ['question','text']

)


url = 'https://medium.com/blog/do-you-have-a-good-iphone-story-f1029e70e302'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'who is author of this blog?','text':docs[0].page_content})

print(result)


