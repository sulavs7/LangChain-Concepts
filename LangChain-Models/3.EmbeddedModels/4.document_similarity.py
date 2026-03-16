# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=300)

documents = [
    "Sachin Tendulkar is considered the greatest batsman of all time, scoring 100 international centuries across Tests and ODIs.",
    "Virat Kohli is known for his aggressive batting style and exceptional fitness, and has scored over 80 international centuries.",
    "MS Dhoni is a legendary wicketkeeper-batsman famous for his calm finishing ability and led India to win the 2011 World Cup.",
    "Rohit Sharma is renowned for his elegant batting and holds the record for the highest individual score in ODIs with 264 runs.",
    "Kapil Dev was India's greatest all-rounder who led India to their first ever World Cup victory in 1983 with his legendary bowling and batting."
]

query = "tell me about rohit kohli"

doc_embeddings= embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index,score = (sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])

print(query)
print(documents[index])
print("similarity score:",score)





