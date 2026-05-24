from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """Agents can be hard to debug and understand. Long context, branching logic, and many tools make it difficult to pinpoint where things went wrong. Tracing breaks each run into a structured timeline of steps so you can see exactly what happened, in what order, and why. """

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0 ,
    separator = ''
)
loader = PyPDFLoader(
    file_path = 'Sulav_Sapkota_CV.pdf'
)

text = loader.load()

# result = splitter.split_text(text)

result = splitter.split_documents(text)
print(result[0].page_content)