from easydict import EasyDict as edict

def get_args():
    args = edict({
        'seed': 123,
        'mode': 'CTDE', # CTDE, Random, IAC, DQN
        'batch_size' : 32, # 32, 256
        'actor_dim' : 64,
        'critic_dim' : 256,
        'train_epoch' : 6500, # 5300 (6500)
        'cuda': True,
        'gamma': 0.98,
        'actor_lr' : 1e-2, # 1e-2
        'critic_lr' : 1e-3, # 1e-3,
        'epsilon' : 0.275, # 0.8, # 1
        'inference_epoch': 100,
        'anneal_epsilon' : 0.00005, # 0.00005, # 0.001
        'min_epsilon' : 0.01,
        # 'td_lambda' : 0.02, # 0.8
        'replay_capacity': 50000, # 10000
        'target_update_cycle' : 20,
        'grad_norm_clip' : 10,
        'k': 2,
        'train_size': 32, # 2000,
        'n_wires' : 4,
    })
    return args

def get_env_info(args,env):
    env_info           = env.get_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_Comm_agents = env_info["n_Comm_agents"]
    args.n_DQN_agents  = env_info["n_DQN_agents"]
    args.n_agents      = env_info["n_agents"]
    args.n_actions     = env_info["n_actions"]
    # args.state_dim     = env_info["state_dim"]
    args.obs_dim       = env_info["obs_dim"]
    # # o = np.hstack([o_idx, o_partial])
    return args