import argparse
from config import AttackConfig
from attack import SemanticCamo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model_name', type=str, default='gpt-4o-2024-08-06', help='Target model name')
    parser.add_argument('--judge_model_name', type=str, default='gpt-4o-2024-08-06', help='Judge model name')
    parser.add_argument('--save_result_dir', type=str, default='./data/gpt4o', help='Directory to save results')
    args = parser.parse_args()

    config = AttackConfig(
        target_model_name=args.target_model_name,
        judge_model_name=args.judge_model_name,
        save_result_dir=args.save_result_dir
    )
    attacker = SemanticCamo(config)
    attacker.START()

