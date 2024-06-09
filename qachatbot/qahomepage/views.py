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

from .readdoc import create_folder
from .loadmodel import load_model

dir_location = create_folder('local_flan_t5_xxl')
model = load_model(
        'google/flan-t5-xxl',
        dir_location
    )

# Create your views here.

def home(request):
    query = request.GET.get('q')
    if query is None:
        query = "who is vignesh?"
    
    
    # data = similar_text('data.txt', query)
    response_text = "sample"
    return HttpResponse(response_text, content_type="text/plain")