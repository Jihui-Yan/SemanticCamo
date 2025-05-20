class AttackConfig:
    def __init__(self, target_model_name='gpt-4o-2024-08-06', judge_model_name='gpt-4o-2024-08-06', save_result_dir='./data/gpt4o'):
        self.target_model_name = target_model_name
        self.judge_model_name = judge_model_name

        self.save_result_dir = save_result_dir


