import numpy as np
import torch
from torch.distributions import Categorical
from policy import DQNActor, CommNetActor, Critic
from copy import deepcopy
import torch.nn.functional as f
import torch.nn as nn
# temperature = 5

class Agent:
    def __init__(self, args, id = None):
        self.id        = id
        self.args      = args
        self.n_actions = args.n_actions
        # self.state_shape = args.state_dim
        self.obs_shape = args.obs_dim
        if id < args.n_Comm_agents : self.actor = CommNetActor(args)
        else                       : self.actor = DQNActor(args)
        self.actor_target = deepcopy(self.actor)
        self.actor.cuda()
        self.actor_target.cuda()
        self.actor_optimizer = torch.optim.Adadelta(self.actor.parameters(), lr=self.args.actor_lr)
        
    def o_preprocessing(self, o):
        if self.id < self.args.n_Comm_agents:
            agent_obs   = np.copy(o)
            IDset       = np.arange(self.args.n_agents)
            logic_IDset = IDset == self.id
            logic_IDset = np.where(logic_IDset == 0)[0]
            logic_IDset = np.hstack([[self.id],logic_IDset])
            agent_obs   = agent_obs[logic_IDset,:]
            agent_obs   = torch.FloatTensor(agent_obs)
        else : 
            agent_obs = torch.FloatTensor(o)[self.id]
        return agent_obs.unsqueeze(0)
    
    def _O_preprocessing(self, O):
        if self.id < self.args.n_Comm_agents:
            agent_obs   = np.copy(O)
            IDset       = np.arange(self.args.n_agents)
            logic_IDset = IDset == self.id
            logic_IDset = np.where(logic_IDset == 0)[0]
            logic_IDset = np.hstack([[self.id],logic_IDset])
            agent_obs   = agent_obs[:,logic_IDset,:]
            agent_obs   = torch.FloatTensor(agent_obs)
        else: 
            agent_obs = torch.FloatTensor(O[:,self.id,:])
        return agent_obs

    def O_preprocessing(self, O):
        agent_obs = torch.FloatTensor(O[:,self.id,:])
        return agent_obs
    
    def _choose_action_from_softmax(self, action_dist, ava, id):
        ava    = torch.FloatTensor(ava).cuda()
        prob   = f.softmax(action_dist, dim=-1) * ava
        # prob   = prob[0][id]
        action = torch.argmax(prob,-1)
        return action
    
    def select_action(self, id, o, ava):
        agent_obs = self.o_preprocessing(o)
        action_dist = self.actor(agent_obs.cuda())
        action  = self._choose_action_from_softmax(action_dist, ava, id)
        return action
    
    # def train(self,O,A,RWD,O_PRIME,critic): ==> Centralized Critic Observes Full Observation Information
    #     bsz       = O.shape[0]
    #     obs       = O
    #     obs_prime = O_PRIME
    #     O         = self._O_preprocessing(O)
    #     O_PRIME   = self._O_preprocessing(O_PRIME)
    #     # O_C       = self.O_preprocessing(obs)
    #     # O_C_PRIME = self.O_preprocessing(obs_prime)
    #     O         = torch.FloatTensor(O).cuda()
    #     O_PRIME   = torch.FloatTensor(O_PRIME).cuda()
    #     O_C       = torch.FloatTensor(obs.reshape(bsz,-1)).cuda()
    #     O_C_PRIME = torch.FloatTensor(obs_prime.reshape(bsz,-1)).cuda()
    #     r         = torch.FloatTensor(RWD).squeeze().cuda().unsqueeze(-1)
    #     A         = torch.tensor(A.reshape(-1,1),dtype=torch.int64).cuda()
    #     Q         = self.actor(O).gather(dim=1, index=A)
    #     TD_TARGET = r + self.args.gamma * critic.critic(O_C_PRIME)
    #     DELTA     = TD_TARGET - critic.critic(O_C)

    #     loss      = - torch.log(Q) * DELTA.detach() + nn.MSELoss()(critic.critic(O_C), TD_TARGET.detach())

    #     self.actor_optimizer.zero_grad()
    #     critic.critic_optimizer.zero_grad()
    #     loss.mean().backward()

    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
    #     self.actor_optimizer.step()

    #     torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), self.args.grad_norm_clip)
    #     critic.critic_optimizer.step()
        
    #     return loss.mean().abs().item()

    def train(self,O,A,RWD,O_PRIME,critic): # ==> Centralized Critic Observes Agentwise Observation Information
        obs       = O
        obs_prime = O_PRIME
        O         = self._O_preprocessing(O)
        O_PRIME   = self._O_preprocessing(O_PRIME)
        O_C       = self.O_preprocessing(obs)
        O_C_PRIME = self.O_preprocessing(obs_prime)
        O         = torch.FloatTensor(O).cuda()
        O_PRIME   = torch.FloatTensor(O_PRIME).cuda()
        O_C       = torch.FloatTensor(O_C).cuda()
        O_C_PRIME = torch.FloatTensor(O_C_PRIME).cuda()
        r         = torch.FloatTensor(RWD).squeeze().cuda().unsqueeze(-1)
        A         = torch.tensor(A.reshape(-1,1),dtype=torch.int64).cuda()
        Q         = self.actor(O).gather(dim=1, index=A)
        if self.args.mode == 'CTDE':
            TD_TARGET = r + self.args.gamma * critic.critic(O_C_PRIME)
            DELTA     = TD_TARGET - critic.critic(O_C)
            loss      = - torch.log(Q) * DELTA.detach() + nn.MSELoss()(critic.critic(O_C), TD_TARGET.detach())

            self.actor_optimizer.zero_grad()
            critic.critic_optimizer.zero_grad()
            loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), self.args.grad_norm_clip)
            critic.critic_optimizer.step()
            critic.critic_optimizer.zero_grad()

            output = loss.mean().abs().item()

        elif self.args.mode == 'IAC' or self.args.mode == 'DDPG':
            # critic_optimizer = torch.optim.Adadelta(critic.parameters(), lr=self.args.actor_lr)
            if   self.args.mode == 'IAC'  : TD_TARGET = r + self.args.gamma * critic.critic(O_C_PRIME)
            elif self.args.mode == 'DDPG' : TD_TARGET = r + self.args.gamma * critic.critic(O_C_PRIME)
            DELTA     = TD_TARGET - critic.critic(O_C)
            loss      = - torch.log(Q) * DELTA.detach() + nn.MSELoss()(critic.critic(O_C), TD_TARGET.detach())

            self.actor_optimizer.zero_grad()
            critic.critic_optimizer.zero_grad()
            loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), self.args.grad_norm_clip)
            critic.critic_optimizer.step()
            critic.critic_optimizer.zero_grad()

            output = loss.mean().abs().item()

        elif self.args.mode == 'DQN' or self.args.mode == 'IQL':
            if   self.args.mode == 'DQN' : Q_TARGET = r + self.args.gamma * self.actor_target(O_PRIME).max(-1).values.detach().unsqueeze(-1)
            elif self.args.mode == 'IQL' : Q_TARGET = r + self.args.gamma * self.actor(O_PRIME).max(-1).values.detach().unsqueeze(-1)
            loss = nn.MSELoss()(Q, Q_TARGET.detach())
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            output = loss.item()

        return output
    

    def soft_update(self):
        for target_param, source_param in zip(self.actor_target.parameters(),
                                              self.actor.parameters()):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * source_param.data)
        
        actor_tmp = deepcopy(self.actor.cpu())
        self.actor_target = None
        self.actor_target = actor_tmp.cuda()

class Critic_V:
    def __init__(self, args):
        self.args             = args
        self.critic           = Critic(args)
        # self.critic_target    = QCritic(args)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.cuda()
        # self.critic_target.cuda()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)