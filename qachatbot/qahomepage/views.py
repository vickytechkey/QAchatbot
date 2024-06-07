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

from .readdoc import model , tokenizer

# Create your views here.

async def home(request):
    query = request.GET.get('q')
    if query is None:
        query = "who is vignesh?"
    data = similar_text('data.txt', query)
    inputs = tokenizer(data, return_tensors="pt")
    outputs = model.generate(**inputs)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return HttpResponse(response_text, content_type="text/plain")