from django.shortcuts import render
from django.http import HttpResponse
import langchain
import langchain_community
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import os
from django.conf import settings
def similar_text(file_path , query):
    file_path = get_file_path(file_path)
    print(file_path)
    loader = TextLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    embedding = HuggingFaceBgeEmbeddings()
    db = FAISS.from_documents(docs , embedding)
    docs_similarity = db.similarity_search(query)
    return docs_similarity

def get_file_path(file_name):
    module_dir = os.path.dirname(__file__)
    folder_path = os.path.join(module_dir, file_name)
    return folder_path
    
    
        
    
    
        
    