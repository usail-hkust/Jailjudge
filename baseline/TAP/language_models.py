import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as genai
import urllib3
from copy import deepcopy

from .config import LLAMA_API_LINK, VICUNA_API_LINK
import emoji
import requests
import json
import re


def remove_emoji(text):
    return emoji.replace_emoji(text, replace="")


def remove_code_blocks(text):
    pattern = r"```.*?```"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def text_process(model_response):
    model_response = remove_emoji(model_response)
    model_response = remove_code_blocks(model_response)
    model_response = model_response.replace("```", "")
    model_response = model_response.replace("#!/bin/bash", "")
    model_response = model_response.replace("os", "")
    model_response = model_response.replace("if __name__ == '__main__':", "")
    return model_response


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self, prompts_list: List, max_n_tokens: int, temperature: float
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([self.tokenizer.encode("}")[1], 29913, 9092, 16675])


class APIModel(LanguageModel):

    API_HOST_LINK = "ADD_LINK"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20

    API_TIMEOUT = 100

    MODEL_API_KEY = os.getenv("MODEL_API_KEY")

    API_HOST_LINK = ""

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """

        output = self.API_ERROR_OUTPUT

        for _ in range(self.API_MAX_RETRY):
            try:

                # Batch generation
                if temperature > 0:
                    # Attack model
                    json1 = {
                        "top_p": top_p,
                        "num_beams": 1,
                        "temperature": temperature,
                        "do_sample": True,
                        "prompt": "",
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }
                else:
                    # Target model
                    json1 = {
                        "top_p": 1,
                        "num_beams": 1,
                        "temperature": 1,  # To prevent warning messages
                        "do_sample": False,
                        "prompt": "",
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }

                    # Do not use extra end-of-string tokens in target mode
                    if "llama" in self.model_name:
                        json1["extra_eos_tokens"] = 0

                if "llama" in self.model_name:
                    # No system prompt for the Llama model
                    assert json1["prompt"] == ""
                    json1["prompt"] = deepcopy(json1["system_prompt"])
                    del json1["system_prompt"]

                resp = urllib3.request(
                    "POST",
                    self.API_HOST_LINK,
                    headers={"Authorization": f"Api-Key {self.MODEL_API_KEY}"},
                    timeout=urllib3.Timeout(self.API_TIMEOUT),
                    json=json1,
                )

                resp_json = resp.json()

                if "vicuna" in self.model_name:
                    if "error" in resp_json:
                        print(self.API_ERROR_OUTPUT)

                    output = resp_json["output"]

                else:
                    output = resp_json

                if type(output) == type([]):
                    output = output[0]

                break
            except Exception as e:
                print("exception!", type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class APIModelLlama7B(APIModel):
    API_HOST_LINK = LLAMA_API_LINK
    MODEL_API_KEY = os.getenv("LLAMA_API_KEY")


class APIModelVicuna13B(APIModel):
    API_HOST_LINK = VICUNA_API_LINK
    MODEL_API_KEY = os.getenv("VICUNA_API_KEY")


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 2000
    API_TIMEOUT = 20

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "YOUR_KEY_HERE"}

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        url = self.url
        headers = self.headers
        # filter text
        for i in range(len(conv)):
            if "content" in conv[i]:
                conv[i]["content"] = text_process(conv[i]["content"])
        for _ in range(self.API_MAX_RETRY):
            # while True:
            try:
                # response = openai.ChatCompletion.create(
                #             model = self.model_name,
                #             messages = conv,
                #             max_tokens = max_n_tokens,
                #             temperature = temperature,
                #             top_p = top_p,
                #             request_timeout = self.API_TIMEOUT,
                #             )
                # output = response["choices"][0]["message"]["content"]
                data = {
                    "model": self.model_name,
                    "messages": conv,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_n_tokens,
                    "request_timeout": self.API_TIMEOUT,
                }
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response_json = response.json()
                print("\n\nresponse_json", response_json)
                output = response_json["choices"][0]["message"]["content"]
                print("output", output)
                break
            except Exception as e:
                output = None
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class PaLM:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        genai.configure(api_key=self.API_KEY)

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of dictionaries,
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = genai.chat(
                    messages=conv, temperature=temperature, top_p=top_p
                )
                output = completion.last

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[: (max_n_tokens * 4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class GeminiPro:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        genai.configure(api_key=self.API_KEY)

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of dictionaries,
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                model = genai.GenerativeModel(self.model_name)
                output = model.generate_content(
                    contents=conv,
                    generation_config=genai.GenerationConfig(
                        candidate_count=1,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_n_tokens,
                    ),
                )

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output.text
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
