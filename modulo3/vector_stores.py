from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

loader = PyPDFDirectoryLoader('./modulo3/contratos')
documentos = loader.load()

print(f'Se cargaron {len(documentos)} documentos desde el directorio')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size  = 1000,
    chunk_overlap = 200
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

vectorstore = Chroma.from_documents(
    docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="./modulo3/chroma_db"
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa Maria Jimenéz Campos?"

resultados = vectorstore.similarity_search(consulta, k=3)
print("Top 3 docuementos ...")
for i, doc in enumerate(resultados,start=1):
    print(f"Contenido {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")