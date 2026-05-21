from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = "claude-3.5-sonet-20241022",temperature=0.5)

result = model.invoke("what is capital of nepal")

print(result.content)