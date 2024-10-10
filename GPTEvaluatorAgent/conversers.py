from GPTEvaluatorAgent.language_models import ChatGPT, Claude, PaLM, HuggingFace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from GPTEvaluatorAgent.config import VICUNA_PATH, LLAMA_PATH, GENERATOR_TEMP, GENERATOR_TOP_P, LLAMA_LORA_MODEL,LORA_WEIGHTS
from peft import PeftModel
import re

def load_generator_model(custom_args):
    # Load attack model and tokenizer
    agent = EvaluatorAgent(model_name = custom_args["base_model"],
                        max_n_tokens = custom_args["max_n_tokens"],
                        max_n_attempts = custom_args["max_n_attempts"],
                        temperature = GENERATOR_TEMP, # init to 1
                        top_p = GENERATOR_TOP_P, # init to 0.9
                        )

    return agent


class GenatorLM():
    """
        Base class for attacker language models.

        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attempts: int,
                 temperature: float,
                 top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attempts = max_n_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_response(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        if len(convs_list[0].messages) == 0:
            #init_message = """### Response: {\"reason\": <reason>your_reason</reason>  \"\",\"score\": <score>your_score</score>\"}"""
            init_message = """### Response: {\"reason\": """
        else:
            init_message = """{\"reason\": \""""

        ERROR_MESSAGE = """{\"reason\": <reason>ERROR</reason>  \"\",\"score\": <score>ERROR</score>\"}"""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for attempt in range(self.max_n_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                       max_n_tokens=self.max_n_tokens,
                                                       temperature=self.temperature,
                                                       top_p=self.top_p
                                                       )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = ERROR_MESSAGE + init_message + full_output
                else:
                    full_output = ERROR_MESSAGE +  full_output

                print(f"""\n{'=' * 80}\n Full Output: {full_output} \n{'=' * 80}\n""")
                reason_answer_pattern = r'<reason>(.*?)</reason>'
                reason_text = re.findall(reason_answer_pattern, full_output)[-1]

                score_answer_pattern = r'<score>(.*?)</score>'
                score_text = re.findall(score_answer_pattern, full_output)[-1]
                extracted_dict = {"reason": reason_text,
                                  "score": score_text}

                if extracted_dict is not None:
                    valid_outputs[orig_index] = extracted_dict
                    convs_list[orig_index].update_last_message(
                        extracted_dict)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attempts} attempts. Terminating.")
        return valid_outputs


class EvaluatorAgent():
    """
        Base class for attacker language models.

        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attempts: int,
                 temperature: float,
                 top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attempts = max_n_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.memory = []
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()


    def simplify_content(self, content):
        # 这里可以添加您的代码来简化内容
        return content
    def update_memory(self, content):
        self.memory.append(content)
    def get_response(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        if len(convs_list[0].messages) == 0:
            #init_message = """### Response: {\"reason\": <reason>your_reason</reason>  \"\",\"score\": <score>your_score</score>\"}"""
            init_message = """### Response: {\"reason\": """
        else:
            init_message = """{\"reason\": \""""

        ERROR_MESSAGE = """{\"reason\": <reason>ERROR</reason>  \"\",\"score\": <score>ERROR</score>\"}"""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for attempt in range(self.max_n_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                       max_n_tokens=self.max_n_tokens,
                                                       temperature=self.temperature,
                                                       top_p=self.top_p
                                                       )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = ERROR_MESSAGE + init_message + full_output

                print("*********full_output******************\n", full_output,"\n*********full_output******************")
                #extracted_dict = common.extract_json(full_output)

                reason_answer_pattern = r'<reason>(.*?)</reason>'
                reason_text = re.findall(reason_answer_pattern, full_output)[-1]

                score_answer_pattern = r'<score>(.*?)</score>'
                score_text = re.findall(score_answer_pattern, full_output)[-1]
                extracted_dict = {"reason": reason_text,
                                  "score": score_text}

                if extracted_dict is not None:
                    valid_outputs[orig_index] = extracted_dict
                    convs_list[orig_index].update_last_message(
                        extracted_dict)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attempts} attempts. Terminating.")
        return valid_outputs


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = ChatGPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    elif model_name in ['llama-2-lora']:
        model = LlamaForCausalLM.from_pretrained(
            LLAMA_LORA_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )
        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        lm = HuggingFace(model_name, model, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":LLAMA_PATH,
            "template":"llama-2"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        },
        "llama-2-lora": {
            "path": LLAMA_PATH,
            "template": "llama-2"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



    