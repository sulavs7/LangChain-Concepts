from langchain_community.document_loaders import PyPDFLoader 

loader = PyPDFLoader(
    file_path = 'Sulav_Sapkota_CV.pdf'
)

docs = loader.load()

print(docs[0].metadata)