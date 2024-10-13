import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch
model_name2model_path ={
    "llamaguard1": "meta-llama/LlamaGuard-7b",
    "llamaguard2": "meta-llama/Meta-Llama-Guard-2-8B",
    "llamaguard3": "meta-llama/Llama-Guard-3-8B",
    "shieldgemma2b": "google/shieldgemma-2b",
    "shieldgemma9b": "google/shieldgemma-9b",
    "ours": "usail-hkust/JailJudge-guard",

}

class Judge_Base:
    def __init__(self, model_name):
        self.model_name = model_name

    def judge(self, setence):
        raise NotImplementedError
    

