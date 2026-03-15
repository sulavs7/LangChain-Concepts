from  langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = 'D:/huggingface_cache' #d drive has space available c is full 

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm)
result = model.invoke("what is the capital of Nepal ? ")

print(result.content)