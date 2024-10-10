import torch
from baseline.GCG import get_nonascii_toks
from baseline.GCG.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from baseline.GCG.minimal_gcg.opt_utils import  get_filtered_cands
from baseline.GCG.minimal_gcg.string_utils import SuffixManager, load_conversation_template
import gc
from GPTEvaluatorAgent.conversers import  load_generator_model
from utils.utils import generate, prefix_check_for_attack_success, agent_check_for_attack_success
from utils.test_utils import test_prefixes



# gradicident-based method
def GCG(model, tokenizer, goal, device, target,args, adv_string_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", num_steps = 500,
    batch_size = 64, topk = 256):

    # llama-2-7b 128
    # llama-2-1b 64


    allow_non_ascii = False  # you can set this to True to use unicode tokens
    template_name  = args.template_name
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
    adv_suffix = adv_string_init
    conv_template = load_conversation_template(template_name)
    user_prompt = goal
    target = target

    suffix_manager = SuffixManager(tokenizer=tokenizer,
                  conv_template=conv_template,
                  instruction=user_prompt,
                  target=target,
                  adv_string=adv_string_init)
    #check_type = args.check_type
    check_type =  "prefixes"
    # load jailbreak evaluate agent
    if args.check_type == "agent":
        custom_args = {
            'base_model': "gpt-4",
            "max_n_tokens": 500,
            "max_n_attempts": 20,
            'n_streams': 1,
            'keep_last_n': 3,
            'n_iterations': 1
        }
        agent = load_generator_model(custom_args)
    for i in range(num_steps):
        #print(f"""\n{'=' * 80}\nIteration: {i}\n{'=' * 80}\n""")
        # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
        input_ids = input_ids.to(device)

        # Step 2. Compute Coordinate Gradient
        coordinate_grad = token_gradients(model,
                                          input_ids,
                                          suffix_manager._control_slice,
                                          suffix_manager._target_slice,
                                          suffix_manager._loss_slice)

        gen_config = model.generation_config
        gen_config.max_new_tokens = 256

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            control_slice_length =  suffix_manager._control_slice.stop - suffix_manager._control_slice.start
            if control_slice_length > 0:

                # Step 3.1 Slice the input to locate the adversarial suffix.
                adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

                # Step 3.2 Randomly sample a batch of replacements.
                new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                                    coordinate_grad,
                                                    batch_size,
                                                    topk=topk,
                                                    temp=1,
                                                    not_allowed_tokens=not_allowed_tokens)

                # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
                # This step is necessary because tokenizers are not invertible
                # so Encode(Decode(tokens)) may produce a different tokenization.
                # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                new_adv_suffix = get_filtered_cands(tokenizer,
                                                    new_adv_suffix_toks,
                                                    filter_cand=True,
                                                    curr_control=adv_suffix)
                input_ids_temp = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
                # Step 3.4 Compute loss on these candidates and take the argmin.
                logits, ids = get_logits(model=model,
                                        tokenizer=tokenizer,
                                        input_ids=input_ids,
                                        control_slice=suffix_manager._control_slice,
                                        test_controls=new_adv_suffix,
                                        return_ids=True)

                losses = target_loss(logits, ids, suffix_manager._target_slice)

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                # Update the running adv_suffix with the best candidate
                adv_suffix = best_new_adv_suffix
            else:
                adv_suffix = adv_string_init

            if check_type == "prefixes":
                is_success = prefix_check_for_attack_success(model,
                                                      tokenizer,
                                                      suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                                                      suffix_manager._assistant_role_slice,
                                                      test_prefixes,
                                                      gen_config= gen_config)
            elif check_type == "agent":

                is_success = agent_check_for_attack_success(model,
                                                            tokenizer,
                                                            suffix_manager.get_input_ids(
                                                                adv_string=adv_suffix).to(device),
                                                            suffix_manager._assistant_role_slice,
                                                            agent,
                                                            custom_args)


        completion = tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice,
                                                gen_config=gen_config))).strip()

        print(f"""\n{'=' * 80}\n  Iteration: {i}\n{'=' * 80}\n""")
        print(f"\n User Query: {goal}")
        print(f"\n Adversarial Prompt: {adv_suffix}")
        print(f"\n Model Response: {completion}")
        print(print(f"\n Length of Model Response: {len(completion)}"))
        print(f"\n Is Success: {is_success}")
        print(f"""\n{'=' * 80}\n Iteration: {i}\n{'=' * 80}\n""")
        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
        # comment this to keep the optimization running for longer (to get a lower loss).
        if is_success:
            break

        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    # Test
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)


    adv_prompt = suffix_manager.tokenizer.decode(suffix_manager.tokenizer.encode(adv_suffix))
    print(f"\nAdv_suffix: {suffix_manager.tokenizer.decode(suffix_manager.tokenizer.encode(adv_suffix))}")

    print(f"\nCompletion: {completion}")
    return adv_prompt, completion,i, is_success

