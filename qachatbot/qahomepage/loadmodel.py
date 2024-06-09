from django.core.cache import cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator,load_checkpoint_and_dispatch
import os

def load_model(model_name , folder_location):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, folder_location)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if os.path.exists(file_path) is  False:
        accelerator = Accelerator()
        accelerator.save_model(model=model, save_directory=file_path,max_shard_size="2GB")
    model = load_checkpoint_and_dispatch(model, checkpoint=file_path, no_split_module_classes=['Block'])
    return model

# def checkformodelcache(cachekey , model_id , folder_location):
#     result = cache.get(cachekey)
#     if result is None:
#         result = load_model(model_id , folder_location)
#         cache.set(cachekey, result, None)
#     return result
        
        