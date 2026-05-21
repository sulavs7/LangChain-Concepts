from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage ,HumanMessage, AIMessage

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()

 
template = PromptTemplate(
    # template = "give name age and city of fictional person \n{format_instruction}",
    template = "give me five facts about {topic} \n{format_instruction}",
    input_variables=['topic'],
    partial_variables= {'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chains = template|model|parser

# final_result = chains.invoke({}) #input_variable is empty but still we have to give a dict so sending blank dict

final_result = chains.invoke({'topic':'black hole'})


print(final_result)
print(type(final_result))

