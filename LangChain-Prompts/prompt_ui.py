from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv 
import streamlit as st 
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text_generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Paper Summarizer")

template = load_prompt('template.json')

paper_input = st.selectbox("Select the Research Paper name",["Select...","Attention Is All You Need","BERT:Pre-Training of Deep Bidirectional Transformers","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Begineer Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length",["short(1-2 paragraphs)","Medium(3-5 paragraphs)","Long(Detailed Explanation)"])


prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
#     chain = template|model
#     result = chain.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })

    result = model.invoke(prompt)
    st.write(result.content)

# summarize attention is all you need paper clearly 