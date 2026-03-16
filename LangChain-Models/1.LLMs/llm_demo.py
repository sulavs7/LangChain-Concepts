
from langchain_openai import OpenAI  # Plain LLM

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("what is the capital of Nepal?")
print(result)  # plain string output


# openai is paid so it gives error of subscription exceeded but if you have the subscription the it will give output like "capital of nepal is kathmandu " ie string in string out in llm