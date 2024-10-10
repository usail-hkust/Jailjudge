from fastchat.model import (
    get_conversation_template
)
from models.language_models import  AgentGPT
from judges.judge_base import JudgeBase
from judges.multi_agent_base import JudgingAgentBase, VotingAgentBase, InferenceAgentBase



class JudgingAgent(JudgingAgentBase):
    def __init__(self, args,judging_agent_id):
        super(JudgingAgent, self).__init__(args,judging_agent_id)
        self.judging_model = AgentGPT(model_name=self.judging_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judging_template_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def reason_score(self, user_input_list, model_response_list):
        convs_list = [self.create_conv(self.get_judging_prompt(user_query, response)) for user_query, response in
                      zip(user_input_list, model_response_list)]
        raw_outputs = self.judging_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature)
        reason_outputs = [self.process_output(raw_output)["reason"] for raw_output in raw_outputs]
        score_outputs = [self.process_output(raw_output)["score"] for raw_output in raw_outputs]
        return reason_outputs, score_outputs





class VotingAgent(VotingAgentBase):
    def __init__(self, args,voting_agent_id):
        super(VotingAgent, self).__init__(args,voting_agent_id)
        self.voting_model = AgentGPT(model_name=self.voting_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.voting_template_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def vote_reason(self, user_input_list, model_response_list,score_list, reason_list):
        convs_list = [self.create_conv(self.get_voting_prompt(user_query, response,score, reason)) for user_query, response,score, reason in
                      zip(user_input_list, model_response_list,score_list, reason_list)]
        raw_outputs = self.voting_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature)
        vote_outputs = [self.process_output(raw_output)["vote"] for raw_output in raw_outputs]
        reason_outputs = [self.process_output(raw_output)["reason"] for raw_output in raw_outputs]
        return vote_outputs, reason_outputs

class InferenceAgent(InferenceAgentBase):
    def __init__(self, args):
        super(InferenceAgent, self).__init__(args)
        self.inference_model = AgentGPT(model_name=self.inference_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.inference_template_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def infer(self,user_input_list, model_output_list, judging_score_list, judging_reason_list, voting_results_list):
        convs_list = [self.create_conv(self.get_inference_prompt(user_input, model_output, judging_score, judging_reason, voting_results)) for user_input, model_output, judging_score, judging_reason, voting_results in
                      zip(user_input_list, model_output_list, judging_score_list, judging_reason_list, voting_results_list)]
        raw_outputs = self.inference_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature)
        judgment_outputs = [self.process_output(raw_output)["judgment"] for raw_output in raw_outputs]
        reason_outputs = [self.process_output(raw_output)["reason"] for raw_output in raw_outputs]
        explanation_outputs = [self.process_output(raw_output)["explanation"] for raw_output in raw_outputs]
        score_outputs = [self.process_output(raw_output)["score"] for raw_output in raw_outputs]
        return judgment_outputs, reason_outputs, explanation_outputs, score_outputs
