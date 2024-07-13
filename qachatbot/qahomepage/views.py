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
from langchain.llms import HuggingFacePipeline
import os
from django.conf import settings
import sentence_transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , pipeline

from .readdoc import similar_text
from .loadmodel import load_model

from profanity_check import predict, predict_prob

# dir_location = create_folder('local_flan_t5_xxl')
file_path = 'flan-t5-small'
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small" , cache_dir=file_path)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small" , cache_dir=file_path)

# Create your views here.

def home(request):
    query = request.GET.get('q')
    if query is None:
        query = "who is vignesh?"
    profanity = predict([query])
    if profanity == 1:
        return HttpResponse("I'm sorry, but I am unable to respond to messages containing inappropriate language. Could you please rephrase your message?", content_type="text/plain")
    data = similar_text('data.txt', query)
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    chain = load_qa_chain(llm ,  chain_type="stuff")
    response_text = chain.run(input_documents =data , question = query )
    return HttpResponse(response_text, content_type="text/plain")