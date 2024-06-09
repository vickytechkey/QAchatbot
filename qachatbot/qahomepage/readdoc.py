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
import sentence_transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "google/flan-t5-xxl"
# local_path = "local_flan_t5_xxl"

# module_dir = os.path.dirname(__file__)
# folder_path = os.path.join(module_dir, local_path)

# tokenizer = AutoTokenizer.from_pretrained(folder_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(folder_path)


def similar_text(file_path , query):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, file_path)
    loader = TextLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    embedding = HuggingFaceBgeEmbeddings()
    db = FAISS.from_documents(docs , embedding)
    docs_similarity = db.similarity_search(query)
    return docs_similarity[0].page_content

def create_folder(folder_name):
    module_dir = os.path.dirname(__file__)
    folder_path = os.path.join(module_dir, folder_name)
    return folder_path
    
    
        
    
    
        
    