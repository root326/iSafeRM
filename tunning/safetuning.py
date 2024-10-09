import omnisafe
from utils import my_logger
from tunning.benchenvs.microservices_env import MicroservicesENV
from tunning.utils import parser_args, get_bench_name

if __name__ == "__main__":
    args = parser_args()
    env_id = ''

    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 64,
            'vector_env_nums': 1,
            'parallel': 1,
            'torch_threads': 16
        },
        'algo_cfgs': {
            'steps_per_epoch': 16,
            # 'cost_limit': 100.0,
            'safety_budget': 100,
            'update_iters': 2,
            'batch_size': 8,
            'unsafe_reward': -1.0
            # 'start_learning_steps': 0,
        },
        'logger_cfgs': {
            # 'use_wandb': False,
            # 'use_tensorboard': True,
            'log_dir': "/home/lilong/iSafeRM/output/run",
            'save_model_freq': 1
        },
    }
    if args.task_name == 'hr':
        hr_list = [
            # "wk20-hr_5causal_1parameters",
            # "wk20-hr_5causal_2parameters",
            # "wk20-hr_5causal_3parameters",
            # "wk50-hr_5causal_3parameters",
            # "wk100-hr_5causal_3parameters",
            "wk150-hr_5causal_3parameters",
            "wk200-hr_5causal_3parameters",
            # "wk20-hr_5causal_4parameters",
            # "wk20-hr_5causal_5parameters",
            # "wk20-hr_3causal",
            # "wk20-hr_4causal",
            # "wk20-hr_5causal",
            # "wk20-hr_6causal",
            # "wk20-hr_7causal",
            # "wk20-hr_8causal"
            # "wk20-hr_5causal_parameters",
            # "wk20-hr_causal_parameters",
        ]
        for i in hr_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()

    if args.task_name == 'sn':
        sn_list = [
            # "wk20-sn_5sp_3param",
            # "wk20-sn_5pba_3param",
            # "wk20-sn_cp_3param",

            # "wk20-sn_svm_3param",

            "wk20-sn_5causal_3parameters",
            # "wk50-sn_5causal_3parameters",
            # "wk100-sn_5causal_3parameters",
            # "wk20-sn_3causal_parameters",
            # "wk20-sn_7causal_5parameters",
            # "wk20-sn_3causal",
            # "wk20-sn_4causal",
            # "wk20-sn_5causal",
            # "wk20-sn_6causal",
            # "wk20-sn_7causal",
            # "wk20-sn_8causal",

        ]
        for i in sn_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()
    if args.task_name == 'mm':
        mm_list = [
            # "wk20-mm_5causal_1parameters",
            # "wk20-mm_5causal_2parameters",
            # "wk20-mm_5causal_3parameters",
            # "wk50-mm_5causal_3parameters",
        ]
        for i in mm_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()
