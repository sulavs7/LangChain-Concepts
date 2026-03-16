from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv 

# chat_template = ChatPromptTemplate(
#     [
#         SystemMessage(content="you are a helpful{domain} expert"),
#         HumanMessage(content='Explain in simple terms, about {topic}')

#     ]
# )

# better way to write mesages 
chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain about {topic} briefly ')
])

prompt = chat_template.invoke({
    'domain':'Cricket',
    'topic':'LBW'
})

print(prompt)