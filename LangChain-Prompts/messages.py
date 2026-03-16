from langchain_core.messages import SystemMessage ,HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)

messages = [SystemMessage(content = 'you are a helpful assistant'),
            HumanMessage(content="tell me about langchain")
            ]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
