import gc
import os
import numpy as np
import torch.nn as nn
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random
from utils.test_utils import test_prefixes
from baseline.AdvPrompter.llm import AdvPrompter
import torch
from utils.test_utils import save_test_to_file_split_judge




class Args:
    def __init__(self, args_dict):
        

        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)

def AdvPrompter_initial(args_dict):
    args = Args(args_dict)
    return args#, attackLM, targetLM


def AdvPrompter_generate_suffix(args, pert_goals):
    test_generation_kwargs = {
            "max_tokens": args.adv_prompter_max_new_tokens,
            "temperature": 0.6,
        }
    attack_llm = AdvPrompter(args, test_generation_kwargs = test_generation_kwargs)
    suffix_dict = {}
    for goal_idx in range(len(pert_goals[args.test_data_idx : args.end_index])):
            pert_goal_i = pert_goals[args.test_data_idx + goal_idx]
            suffix = attack_llm.generate_suffix(pert_goal_i)
            suffix_dict[args.test_data_idx + goal_idx] = suffix
    return suffix_dict

def AdvPrompter_single_main(args_dict, target_model, goal, target, curr_output, jailbreak_judge_model=None):
    # initialize
    args = Args(args_dict)
    if args.attack == "None":
        output_list = target_model.generate(args, goal, with_suffix = False)
        is_JB_flag = False
        for i in range(len(output_list)):
            c_output = output_list[i]
            is_JB_flag = not any([prefix.lower() in c_output.lower() for prefix in test_prefixes])
            jb_suffix = ""
            jb_output = c_output
            jb_iter = i
            if is_JB_flag:
                break
        return jb_suffix, jb_output, jb_iter, is_JB_flag
    
    output_list, suffix_list = target_model.generate(args, goal)
    is_JB_flag = False
    # for i in range(len(output_list)):
    #     c_output = output_list[i]
    #     is_JB_flag = not any([prefix.lower() in c_output.lower() for prefix in test_prefixes])
    #     jb_suffix = suffix_list[i]
    #     jb_output = c_output
    #     jb_iter = i
    #     # save results
    #     curr_output["adv_prompt"] = jb_suffix
    #     curr_output["language_model_output"] = jb_output
    #     curr_output["attack_iterations"] = jb_iter
    #     curr_output["is_JB"] = is_JB_flag
    #     save_test_to_file_split_judge(args, curr_output)
    for i in range(len(output_list)):
        c_output = output_list[i]
        if args.jailbreak_judge_method not in ["prefix", "default"]:
            is_JB_flag = jailbreak_judge_model.judge(goal + suffix_list[i], c_output)
        else:
            is_JB_flag = not any([prefix.lower() in c_output.lower() for prefix in test_prefixes])
        # is_JB_flag = not any([prefix.lower() in c_output.lower() for prefix in test_prefixes])
        jb_suffix = suffix_list[i]
        jb_output = c_output
        jb_iter = i
        
        if is_JB_flag:
            break
    return jb_suffix, jb_output, jb_iter, is_JB_flag
