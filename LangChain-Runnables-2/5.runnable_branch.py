from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda,RunnableBranch


from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate me a report on topic {topic}',
    input_variables = ['topic']

)


prompt2 = PromptTemplate(
    template = 'Generate me a summary on this text {text}',
    input_variables = ['text']

)


report_gen_chain = RunnableSequence(prompt1 , model , parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 200, RunnableSequence(prompt2 , model , parser))
    ,
    RunnablePassthrough()

)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(result)