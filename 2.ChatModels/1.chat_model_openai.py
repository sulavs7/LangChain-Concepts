from langchain_openai import ChatOpenAI

from langchain_groq import  ChatGroq 
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model ="openai/gpt4" , temperature=0.8)
model = ChatGroq(model ="openai/gpt-oss-20b" , temperature=0.8)

result = model.invoke("write five line poem on current status of nepal  " )

print(result.content)


