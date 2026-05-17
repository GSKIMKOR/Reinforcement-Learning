from argument import get_args, get_env_info
from environmentComm import Environment
from memory import ReplayBuffer, get_experience
import torch
import copy
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agent import Agent, Critic_V
import json

DICT = {
    'Total':[],
    'Agent1':[],
    'Agent2':[],
    'Agent3':[],
    'Agent4':[],
    'Agent5':[],
    'Agent6':[],
    'Agent7':[],
    'Agent8':[],
    'Agent9':[],
    'Agent10':[],
}

env = Environment()
args = get_args()
args = get_env_info(args, env)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def detach_cuda(Agents):
    for agent in Agents:
        agent.actor = agent.actor.cpu()
        agent.actor_target = agent.actor_target.cpu()

def make_cuda(Agents):
    for agent in Agents:
        agent.actor = agent.actor.cuda()
        agent.actor_target = agent.actor_target.cuda()

Agents = [Agent(args = args, id = i) for i in range(args.n_agents)]
if args.mode == 'IAC': Critics = [Critic_V(args=args) for _ in range(args.n_agents)]
else : Central_V = Critic_V(args)
replay = ReplayBuffer(args)
reward_max = -10000
today = datetime.today().strftime("%m%d%H%M%S")
info  = f'{args.mode}_Comm{args.n_Comm_agents}_DNN{args.n_DQN_agents}_lr{args.actor_lr}'
writer = SummaryWriter(f'runs/{today}+{info}')
x_COMM, y_COMM, x_DNN, y_DNN = None, None, None, None
for epoch in range(args.train_epoch+1):
    env.reset()
    o       = env.o # n_agents, obs_dim
    o_prime = env.o_prime
    ava     = env.get_available_action()
    TB_RWD  = np.zeros(args.n_agents)
    TB_EN   = np.zeros(args.n_agents)
    # replay.buffer.clear()
    takeoff = None
    service = np.zeros([args.episode_limit+1, args.n_agents+1])
    energy  = np.zeros([args.episode_limit+1, args.n_agents])
    # replay.initialize()
    for t in range(args.episode_limit+1):
        o = o_prime.copy()
        actions, u_onehot = [], np.zeros((args.n_agents,args.n_actions))
        for i in range(args.n_agents):
            coin = np.random.rand()
            if args.mode != 'Random':
                if coin >= args.epsilon:
                    action = Agents[i].select_action(id=i, o=o, ava=ava[i]).squeeze().data.cpu().numpy()
                else:
                    while(True):
                        action = np.random.randint(low=0,high=args.n_actions)
                        if ava[i][action] == 1: break
            else:
                while(True):
                    coin = np.random.rand()
                    action = np.random.randint(low=0,high=args.n_actions)
                    if ava[i][action] == 1: break
            actions.append(action)
        actions = np.array(actions)
        o, rewards, o_prime, ava, takeoff, x_COMM, y_COMM, x_DNN, y_DNN = env.step(actions, t, epoch, today, info)
        # experience = get_experience(o, actions, rewards.sum(), o_prime)
        experience = get_experience(o, actions, rewards, o_prime)
        replay.push(experience)
        TB_RWD += rewards
        _, TB_EN = env.utility.get_utils_info()
        # For every time step plotting
        for i in range(args.n_agents+1) : service[t][i] = env.utility.SERVICED[i]
        for i in range(args.n_agents)   :  energy[t][i] = TB_EN[i]
    
    # [1] Sampling Batch Experiences
    args.epsilon -= args.anneal_epsilon
    args.epsilon  = max(args.epsilon, args.min_epsilon)

    if args.mode != 'Random':
        if replay.size() > args.train_size:
            samples = replay.sample()
            O = samples['O']
            ACTIONS = samples['A']
            REWARDS = samples['REWARDS']
            O_PRIME = samples['O_PRIME']
            ACTOR_LOSS  = []
            # CRITIC_LOSS = []
            
            for i in range(args.n_agents):
                # actor_loss = Agents[i].train(O,ACTIONS[:,i],REWARDS,O_PRIME,Central_V)
                if args.mode == 'IAC': actor_loss = Agents[i].train(O,ACTIONS[:,i],REWARDS[:,i],O_PRIME,Critics[i])
                else:                  actor_loss = Agents[i].train(O,ACTIONS[:,i],REWARDS[:,i],O_PRIME,Central_V)
                ACTOR_LOSS.append(actor_loss)
                # CRITIC_LOSS.append(critic_loss)
            ACTOR_LOSS  = np.array(ACTOR_LOSS)
            # CRITIC_LOSS = np.array(CRITIC_LOSS)

            if args.mode == 'DQN' or args.mode == 'DDPG':
                if epoch != 0 and epoch % args.target_update_cycle == 0:
                    detach_cuda(Agents)
                    for agent in Agents:
                        agent.actor_target.load_state_dict(agent.actor.state_dict())
                    make_cuda(Agents)

            # writer.add_scalars(f'loss/critic', {
            #     'critic': CRITIC_LOSS.sum() / args.n_agents
            # }, epoch)

            writer.add_scalars(f'loss/actor', {
                'total' : ACTOR_LOSS.sum(),
                'agent1': ACTOR_LOSS[0],
                'agent2': ACTOR_LOSS[1],
                'agent3': ACTOR_LOSS[2],
                'agent4': ACTOR_LOSS[3],
                'agent5': ACTOR_LOSS[4],
                'agent6': ACTOR_LOSS[5],
                'agent7': ACTOR_LOSS[6],
                'agent8': ACTOR_LOSS[7],
                'agent9': ACTOR_LOSS[8],
                'agent10': ACTOR_LOSS[9],
            }, epoch)

    for i in range(args.n_agents):
        total = 0
        for j in range(5):
            if takeoff[i][j] > 0:
                total += 1
        writer.add_scalars(f'takeoff/agent{i}',{
                'numVertiport': total,
                'vertiport1': takeoff[i][0],
                'vertiport2': takeoff[i][1],
                'vertiport3': takeoff[i][2],
                'vertiport4': takeoff[i][3],
                'vertiport5': takeoff[i][4]
            }, epoch)

    reward_max = max(reward_max, TB_RWD.sum())
    writer.add_scalars(f'Reward/reward', {
            'total' : TB_RWD.sum(),
            'agent1': TB_RWD[0],
            'agent2': TB_RWD[1],
            'agent3': TB_RWD[2],
            'agent4': TB_RWD[3],
            'agent5': TB_RWD[4],
            'agent6': TB_RWD[5],
            'agent7': TB_RWD[6],
            'agent8': TB_RWD[7],
            'agent9': TB_RWD[8],
            'agent10': TB_RWD[9],
        }, epoch)

    DICT['Total'].append(TB_RWD.sum())
    DICT['Agent1'].append(TB_RWD[0])
    DICT['Agent2'].append(TB_RWD[1])
    DICT['Agent3'].append(TB_RWD[2])
    DICT['Agent4'].append(TB_RWD[3])
    DICT['Agent5'].append(TB_RWD[4])
    DICT['Agent6'].append(TB_RWD[5])
    DICT['Agent7'].append(TB_RWD[6])
    DICT['Agent8'].append(TB_RWD[7])
    DICT['Agent9'].append(TB_RWD[8])
    DICT['Agent10'].append(TB_RWD[9])

    pd.DataFrame(DICT['Total']).to_csv(f'./runs/{today}+{info}/Total.csv')
    pd.DataFrame(DICT['Agent1']).to_csv(f'./runs/{today}+{info}/Agent1.csv')
    pd.DataFrame(DICT['Agent2']).to_csv(f'./runs/{today}+{info}/Agent2.csv')
    pd.DataFrame(DICT['Agent3']).to_csv(f'./runs/{today}+{info}/Agent3.csv')
    pd.DataFrame(DICT['Agent4']).to_csv(f'./runs/{today}+{info}/Agent4.csv')
    pd.DataFrame(DICT['Agent5']).to_csv(f'./runs/{today}+{info}/Agent5.csv')
    pd.DataFrame(DICT['Agent6']).to_csv(f'./runs/{today}+{info}/Agent6.csv')
    pd.DataFrame(DICT['Agent7']).to_csv(f'./runs/{today}+{info}/Agent7.csv')
    pd.DataFrame(DICT['Agent8']).to_csv(f'./runs/{today}+{info}/Agent8.csv')
    pd.DataFrame(DICT['Agent9']).to_csv(f'./runs/{today}+{info}/Agent9.csv')
    pd.DataFrame(DICT['Agent10']).to_csv(f'./runs/{today}+{info}/Agent10.csv')

    writer.add_scalars(f'Utility/num_serviced', {
            'total' : env.utility.SERVICED[0],
            'agent1': env.utility.SERVICED[1],
            'agent2': env.utility.SERVICED[2],
            'agent3': env.utility.SERVICED[3],
            'agent4': env.utility.SERVICED[4],
            'agent5': env.utility.SERVICED[5],
            'agent6': env.utility.SERVICED[6],
            'agent7': env.utility.SERVICED[7],
            'agent8': env.utility.SERVICED[8],
            'agent9': env.utility.SERVICED[9],
            'agent10': env.utility.SERVICED[10],
        }, epoch)

    if (epoch % 500 == 0) or (( reward_max == TB_RWD.sum() ) and epoch > 3000) :
        for t in range(args.episode_limit+1):
            writer.add_scalars(f'serviced_num_user/EP{epoch+1}', {
            'total' : service[t][0],
            'agent1': service[t][1],
            'agent2': service[t][2],
            'agent3': service[t][3],
            'agent4': service[t][4],
            'agent5': service[t][5],
            'agent6': service[t][6],
            'agent7': service[t][7],
            'agent8': service[t][8],
            'agent9': service[t][9],
            'agent10': service[t][10],
            }, t)
            
            writer.add_scalars(f'energy_consumption/EP{epoch+1}', {
                'agent1': energy[t][0],
                'agent2': energy[t][1],
                'agent3': energy[t][2],
                'agent4': energy[t][3],
                'agent5': energy[t][4],
                'agent6': energy[t][5],
                'agent7': energy[t][6],
                'agent8': energy[t][7],
                'agent9': energy[t][8],
                'agent10': energy[t][9],
            }, t)

        if args.mode == 'IAC':
            for i in range(args.n_agents):
                torch.save({
                            'actor': Agents[i].actor.state_dict(),
                            'actor_optimizer': Agents[i].actor_optimizer.state_dict()
                            }, f'./runs/{today}+{info}/actor{i}_epoch{epoch}_{TB_RWD.sum()}.tar')
                torch.save({
                            'critic': Critics[i].critic.state_dict(),
                            'critic_optimizer': Critics[i].critic_optimizer.state_dict()
                            }, f'./runs/{today}+{info}/critic{i}_epoch{epoch}_{TB_RWD.sum()}.tar')
        else: 
            for i in range(args.n_agents):
                torch.save({
                            'actor': Agents[i].actor.state_dict(),
                            'actor_optimizer': Agents[i].actor_optimizer.state_dict()
                            }, f'./runs/{today}+{info}/actor{i}_epoch{epoch}_{TB_RWD.sum()}.tar')
            
            torch.save({
                        'critic': Central_V.critic.state_dict(),
                        'critic_optimizer': Central_V.critic_optimizer.state_dict()
                        }, f'./runs/{today}+{info}/critic_epoch{epoch}_{TB_RWD.sum()}.tar')
            

        if env.numCommAgent > 0 :

            with open(f'./runs/{today}+{info}/x_position_CommNet_{epoch}.csv', 'w') as f:
                x_COMM = x_COMM.tolist()
                json.dump(x_COMM, f)

            with open(f'./runs/{today}+{info}/y_position_CommNet_{epoch}.csv', 'w') as f:
                y_COMM = y_COMM.tolist()
                json.dump(y_COMM, f)

        if env.numDQNAgent > 0 :

            with open(f'./runs/{today}+{info}/x_position_DNN_{epoch}.csv', 'w') as f:
                x_DNN = x_DNN.tolist()
                json.dump(x_DNN, f)

            with open(f'./runs/{today}+{info}/y_position_DNN_{epoch}.csv', 'w') as f:
                y_DNN = y_DNN.tolist()
                json.dump(y_DNN, f)

