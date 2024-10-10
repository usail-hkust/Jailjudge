from judge_agent.llama_guard import LlamaGuard1, LlamaGuard2, LlamaGuard3
from judge_agent.shieldgemma import ShieldGemma
from judge_agent.base import Judge_Base, model_name2model_path
from judge_agent.jbjudge import JbJudge

def init_judges(args):
    if args.jailbreak_judge_method == "llamaguard1":
        judges = LlamaGuard1(args.jailbreak_judge_method)
    elif args.jailbreak_judge_method == "llamaguard2":
        judges = LlamaGuard2(args.jailbreak_judge_method)
    elif args.jailbreak_judge_method == "llamaguard3":
        judges = LlamaGuard3(args.jailbreak_judge_method)
    elif args.jailbreak_judge_method == "shieldgemma2b":
        judges = ShieldGemma(args.jailbreak_judge_method)
    elif args.jailbreak_judge_method == "shieldgemma9b":
        judges = ShieldGemma(args.jailbreak_judge_method)
    elif args.jailbreak_judge_method == "ours":
        judges = JbJudge(args.jailbreak_judge_method)
    else:
        raise ValueError(f"Unknown model {args.jailbreak_judge_method}")
    return judges