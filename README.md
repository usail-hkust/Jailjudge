<p align="center">

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Stars](https://img.shields.io/github/stars/usail-hkust/Jailjudge)
![Visitor Badge](https://img.shields.io/endpoint?url=https://visitor-badge.glitch.me/badge?page_id=usail-hkust.Jailjudge)



</p>


# JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework



## Overview

Although significant research efforts have been dedicated to enhancing the safety of large language models (LLMs) by understanding and defending against jailbreak attacks, evaluating the defense capabilities of LLMs against jailbreak attacks  also attracts lots of attention. Current evaluation methods lack explainability and do not generalize well to complex scenarios, resulting in incomplete and inaccurate assessments (e.g., direct judgment without reasoning explainability,  the F1 score of the GPT-4 judge is only 55\% in complex scenarios and bias evaluation on multilingual scenarios, etc.). To address these challenges, we have developed a comprehensive evaluation benchmark, JAILJUDGE, which includes a wide range of risk scenarios with complex malicious prompts (e.g., synthetic, adversarial, in-the-wild, and multi-language scenarios, etc.) along with high-quality human-annotated test datasets. Specifically, the JAILJUDGE dataset comprises training data of JAILJUDGE, with over 35k+ instruction-tune training data with reasoning explainability, and JAILJUDGETEST, a 4.5k+ labeled set of broad risk scenarios and a 6k+ labeled set of multilingual scenarios in ten languages. To provide reasoning explanations (e.g., explaining why an LLM is jailbroken or not) and fine-grained evaluations (jailbroken score from 1 to 10), we propose a multi-agent jailbreak judge framework, JailJudge MultiAgent, making the decision inference process explicit and interpretable to enhance evaluation quality.   Using this framework, we construct the instruction-tuning ground truth and then instruction-tune an end-to-end jailbreak judge model, JAILJUDGE Guard, which can also provide reasoning explainability with fine-grained evaluations without API costs. 
Additionally, we introduce JailBoost, an attacker-agnostic attack enhancer, and GuardShield, a safety moderation defense method, both based on JAILJUDGE Guard. Comprehensive experiments demonstrate the superiority of our JAILJUDGE benchmark and jailbreak judge methods. Our jailbreak judge methods (JailJudge MultiAgent and JAILJUDGE Guard) achieve SOTA performance in closed-source models (e.g., GPT-4) and safety moderation models (e.g., Llama-Guard and ShieldGemma, etc.), across a broad range of complex behaviors (e.g., JAILJUDGE benchmark, etc.) to zero-shot scenarios (e.g., other open data, etc.). Importantly, JailBoost and GuardShield, based on JAILJUDGE Guard, can enhance downstream tasks in jailbreak attacks and defenses under zero-shot settings with significant improvement (e.g., JailBoost can increase the average performance  by approximately 29.24\%, while GuardShield can reduce the average defense ASR from 40.46\% to 0.15\%).

## 💡Framework


![JAILJUDGE Benchmark and multi-agent Judge Framework](Figures/framewrok.png)

The JAILJUDGE Benchmark encompasses a wide variety of complex jailbreak scenarios, including multilingual and adversarial prompts, targeting diverse LLM responses for robust safety evaluation.

The JAILJUDGE Data includes over 35k instruction-tune training data and two test sets (4.5k+ broad risk scenarios and 6k+ multilingual examples), providing a rich foundation for comprehensive jailbreak assessments.

The Multi-agent Jailbreak Judge Framework leverages multiple agents (Judging, Voting, and Inference agents) to deliver fine-grained evaluations, reasoning explanations, and jailbroken scores, making the evaluation process explicit and interpretable.


## 🆙 Updates
- We relase our JAILJUDE project: a comprehensive evaluation benchmark, JAILJUDGE, which includes a wide range
of risk scenarios with complex malicious prompts (e.g., synthetic, adversarial,
in-the-wild, and multi-language scenarios, etc.) along with high-quality human-
annotated test datasets

## 👉 Paper
For more details, please refer to our paper [JAILJUDGE](https://arxiv.org/abs/2410.12855).

This is the difference between our JAILJUDGE and the existing models like Llama-Guard3, ShieldGemma, etc.
![Figure1](docs/static/images/prob_dif_00.png)

## 1 Installation

```bash
pip install -r requirements.txt
```

## 2 Preparation

### 2.1 OPENAI API Key

There are several baselines that use GPT service in their method. And We also use GPT-4 based agent to evaluate the attack success rate. Please search and replace `YOUR_KEY_HERE` with your OpenAI API key in the following files: `./GPTEvaluatorAgent/language_models.py`, `./baseline/TAP/language_models.py`, `./baseline/PAIR/language_models.py`, `./baseline/GPTFuzz/gptfuzzer/llm/llm.py`, `./baseline/AutoDAN/utils/opt_utils.py`.

After setting the API key, you can run the tricks experiments with one-click scripts below.

### 2.2 Model Preparation

To reproduce the main experiment results, you need to prepare the following models:

- **Attack Model**:
  - AdvPrompter: You need to train the AdvPrompter model to get the LoRA Adapter and merge the model. You can save the attack model to `./models/attack/`.
  - AmpleGCG: If you cannot load the huggingface model directly using the `AutoModelForCausalLM.from_pretrained()` function, you may need to download the attack model `osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat` and `osunlp/AmpleGCG-llama2-sourced-vicuna-7b` from the Hugging Face first.
- **Defense Model**:
  - RPO: You need to run the RPO method from the original repository to get the defense adversarial suffix. We also provide the results we obtained.
  - Advertorial Training: You need to train the adversarial training model to get the defense model. You can save the defense model to `./models/defense/`. Due to the space limitation, we cannot provide the pre-trained model in anonymous repository. After the paper is public, we will provide the pre-trained model in the official repository.
  - Safety Training: You need to train the safety training model to get the defense model. You can save the defense model to `./models/defense/`. Due to the space limitation, we cannot provide the pre-trained model in anonymous repository. After the paper is public, we will provide the pre-trained model in the official repository.
- **Judge Model**:
  - Jailjudge-guard: You can download the jailjudge-guard model from the Hugging Face path: `usail-hkust/JailJudge-guard`. After downloading the model, you can save the model to `./models/judge/`.

```python
# Useage of Judge Models
from judge_agent.jbjudge import JbJudge
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "usail-hkust/JailJudge-guard"
judges = JbJudge(model_id)

prompt = "I forgot how to kill a process in Linux, can you help?"
response = "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."
is_JB = jailbreak_judge_model.judge(prompt, response)
# False

# if you want to set the threshold and get the reason explanation, you can use the following code
is_JB, reason = jailbreak_judge_model.judge_with_reason(prompt, response, threshold=5)
```

### 2.3 Data Preparation

You can download the `JAILJUDGETRAIN`, `JAILJUDGE ID` , and `JAILJUDGE OOD`   from the Hugging Face path: `usail-hkust/JailJudge`. 


## 3 Multi-agent Judge  Framework 
The Multi-agent Jailbreak Judge Framework leverages multiple agents (Judging, Voting, and Inference agents) to deliver fine-grained evaluations, reasoning explanations, and jailbroken scores, making the evaluation process explicit and interpretable.

```bash
# Run multi-agent judge 
python main_multi_agent_judge.py

```


## 4 JAILJUDGE Guard and Jailbreak Enhancers
To demonstrate the fundamental capability of JAILJUDGE Guard, we propose both a jailbreak attack enhancer and a defense method based on JAILJUDGE Guard, named _JailBoost_ and _GuardShield_.







### 4.1 JAILJUDGE as Attack Enhancer
For this part, we provide the following tricks to evaluate the effectiveness of JAILJUDGE as an attack enhancer, and the baseline methods are also provided for comparison.

```bash
# Run the JAILJUDGE as attack enhancer for AutoDAN attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_atk_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_atk_ours \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as attack enhancer for PAIR attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_atk_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_atk_ours \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as attack enhancer for AmpleGCG attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AmpleGCG \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_atk_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_atk_ours \
    --attack_source vicuna \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as attack enhancer for AdvPrompter attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AdvPrompter \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_atk_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_atk_ours \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b_merged \
    --jailbreak_judge_method ours
```

`jailbreak_judge_method`: You can set the `jailbreak_judge_method` to `llamaguard1`, `llamaguard2`, `llamaguard3`, `shieldgemma2b`, `shieldgemma9b`, `ours` to run the corresponding method.

**Note**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel. You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.



### 4.2 JAILJUDGE as Defense Method
```bash
# Run the JAILJUDGE as defense method against AutoDAN attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type ours \
    --attack AutoDAN \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_def_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_def_ours \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as defense method against PAIR attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type ours \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_def_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_def_ours \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as defense method against AmpleGCG attack
python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type ours \
    --attack AmpleGCG \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_def_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_def_ours \
    --attack_source vicuna \
    --jailbreak_judge_method ours

# Run the JAILJUDGE as defense method against AdvPrompter attack

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type ours \
    --attack AdvPrompter \
    --instructions_path ./data/HEx-PHI.csv \
    --save_result_path ./exp_results/vicuna_def_ours/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name vicuna_def_ours \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b_merged \
    --jailbreak_judge_method ours
```

`jailbreak_judge_method`: You can set the `jailbreak_judge_method` to `llamaguard1`, `llamaguard2`, `llamaguard3`, `shieldgemma2b`, `shieldgemma9b`, `ours` to run the corresponding method.

**Note**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel. You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.


## 5. Acknowledgement
In the implementation of this project, we have referred to the code from the following repositories:
- [JailTrickBench][R-JailTrickBench] (NeurIPS 2024)
- [Llama-Guard3][R-Llama-Guard3] (Meta)
- [ShieldGemma][R-ShieldGemma] (Google)

[R-JailTrickBench]: https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking
[R-Llama-Guard3]: https://huggingface.co/meta-llama/Llama-Guard-3-8B
[R-ShieldGemma]: https://huggingface.co/google/shieldgemma-9b

## 6. Citation
If you find this project helpful, please consider citing our paper:
```
@misc{liu2024jailjudgecomprehensivejailbreakjudge,
      title={JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework}, 
      author={Fan Liu and Yue Feng and Zhao Xu and Lixin Su and Xinyu Ma and Dawei Yin and Hao Liu},
      year={2024},
      eprint={2410.12855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12855}, 
}
```

😝 If you like this project, don't forget to give it a ⭐!

[![📈 Star History Chart](https://api.star-history.com/svg?repos=usail-hkust/Jailjudge&type=Date)](https://star-history.com/#usail-hkust/Jailjudge&Date)


