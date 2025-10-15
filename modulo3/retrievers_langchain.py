from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="./modulo3/chroma_db"
)

# Instanciar un retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

consulta = "¿Dónde se encuentra el local del contrato en el que participa Maria Jimenéz Campos?"

resultados = retriever.invoke(consulta)
print("Top 3 docuementos ...")
for i, doc in enumerate(resultados,start=1):
    print(f"Contenido {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")