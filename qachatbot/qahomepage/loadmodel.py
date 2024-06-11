from django.core.cache import cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , BitsAndBytesConfig
from accelerate import Accelerator,load_checkpoint_and_dispatch
import os
import torch

def load_model(local_model):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small" , cache_dir=file_path)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small" , cache_dir=file_path)
    return [model , tokenizer]
        
        