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
            'safety_budget': 100,
            'update_iters': 2,
            'batch_size': 8,
            'unsafe_reward': -1.0
        },
        'logger_cfgs': {
            'log_dir': "/home/XXXX-1/iSafeRM/output/run",
            'save_model_freq': 1
        },
    }
    if args.task_name == 'hr':
        hr_list = [
            "wk150-hr_5causal_3parameters",
            "wk200-hr_5causal_3parameters",
        ]
        for i in hr_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()

    if args.task_name == 'sn':
        sn_list = [

            "wk20-sn_5causal_3parameters",

        ]
        for i in sn_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()
    if args.task_name == 'mm':
        mm_list = [
        ]
        for i in mm_list:
            env_id = i
            agent = omnisafe.Agent('PPOSaute', env_id, custom_cfgs=custom_cfgs)
            agent.learn()
