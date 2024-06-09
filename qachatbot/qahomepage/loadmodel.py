from django.core.cache import cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_model(model_name , folder_location):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name ,cache_dir=folder_location)
    return model

def checkformodelcache(cachekey , model_id , folder_location):
    result = cache.get(cachekey)
    if result is None:
        result = load_model(model_id , folder_location)
        cache.set(cachekey, result, None)
    return result
        
        