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


# Create your views here.

def home(request):
    query = request.GET.get('q')
    if query is None:
        query = "who is vignesh?"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'data.txt')
    print(file_path)
    loader = TextLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    embedding = HuggingFaceBgeEmbeddings()
    db = FAISS.from_documents(docs , embedding)
    docs_similarity = db.similarity_search(query)
    llm = HuggingFaceHub(repo_id='google/flan-t5-xxl' , model_kwargs={"temprature":0.8 , "max_length":520})
    chain = load_qa_chain(llm ,  chain_type="stuff")
    response_text = chain.run(input_documents =docs_similarity , question = query )
    return HttpResponse(response_text, content_type="text/plain")