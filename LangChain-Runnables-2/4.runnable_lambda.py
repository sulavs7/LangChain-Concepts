from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv

load_dotenv()
passthrough = RunnablePassthrough()



prompt1 = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables = {'topic'}
)

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

def word_count(text):
    return(len(text.split()))


joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
    #'word_count':RunnableLambda(lambda x : len(x.split) )
}
)

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'AI'})
print(result)
