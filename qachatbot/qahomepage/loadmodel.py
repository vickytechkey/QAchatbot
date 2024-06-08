from django.core.cache import cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_model(model_name , folder_name):
    module_dir = os.path.dirname(__file__)
    folder_path = os.path.join(module_dir, folder_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name ,cache_dir=folder_name)
    return model

def checkformodelcache(cachekey , model_id , folder_name):
    result = cache.get(cachekey)
    if result is None:
        result = load_model(model_id , folder_name)
        cache.set(cache_key, result, None)
    return result
        
        