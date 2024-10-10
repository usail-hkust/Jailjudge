import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
import google.generativeai as palm
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import emoji
import requests
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm

def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace='')
    model_response = model_response.replace("```", "")
    return  model_response



    

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")




class LocalVLLM(LLM):
    def __init__(self,
                 args,
                 gpu_memory_utilization=0.6,
                 test_generation_kwargs=None,
                 suffix_dict={},
                 ):
        super().__init__()
        self.model_path = args.target_model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",)
        
        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.suffix_dict = suffix_dict
        self.batch_size = 50
    def generate(self,args, pert_goal, with_suffix=True):
        outputs = []
        if with_suffix:
            suffix = self.suffix_dict[args.test_data_idx]
            for bi in tqdm(range(0, len(suffix), self.batch_size), desc="Batch Generation"):
                batch_suffix = suffix[bi:bi+self.batch_size]
                batch_pert_goals = [pert_goal + s for s in batch_suffix]
                
                response = self.model.generate(prompts=batch_pert_goals, sampling_params = self.sampling_params)
                batch_response = [r.outputs[0].text for r in response]
                outputs.extend(batch_response)
            return outputs, suffix
        else:
            batch_pert_goals = [pert_goal]
            response = self.model.generate(prompts=batch_pert_goals, sampling_params = self.sampling_params)
            batch_response = [r.outputs[0].text for r in response]
            outputs.extend(batch_response)
        return outputs
    


class AdvPrompter:
    def __init__(self,
                 args,
                 gpu_memory_utilization=0.6,
                 test_generation_kwargs=None,
                 suffix_dict={},
                 ):
        super().__init__()
        self.model_path = args.adv_prompter_model_path
        self.batch_size = 50
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",max_num_seqs=self.batch_size)
        
        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.max_iters = args.adv_prompter_max_iters
        
    def generate_suffix(self, goal):
        prompts = [goal] * self.max_iters
        response = self.model.generate(
            prompts=prompts, sampling_params=self.sampling_params
        )
        output = [r.outputs[0].text for r in response]
        return output
        