from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage ,HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)

# chat_history = []
messages=[SystemMessage(content='you are a helpful ai assistant')]

while True:
    user_input = input('You: ')
    # chat_history.append(user_input)
    messages.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break 
    # result = model.invoke(chat_history)
    result = model.invoke(messages)
    # chat_history.append(result.content)
    messages.append(AIMessage(content=result.content))
    print("AI: ",result.content)

# print(chat_history)
print(messages)
