import numpy as np
import pandas as pd
import math
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
import io
import PIL.Image
from torchvision.transforms import ToTensor

os.chdir(os.path.dirname(__file__)) # 실행 경로를 현재 .py 파일이 위치한 곳(서버의 위치)으로 변경

EP_LEN      = 60
N_VERTIPORT = 5
N_USER      = 200
N_COMMAGENT = 10
N_DQNAGENT  = 0
GRID        = 32000 # [m]
OBSERVABLE  = GRID * 3 # GRID * 1.5 / 3
H_MAX       = 600   # [m] ==> UAM이 horizontal moving하는 altitude에 해당
H_MIN       = 0     # [m] ==> Vertiport의 altitude에 해당
VELOCITY    = 4425  # [m/min] 
MTOW        = 4

font = {'size'   :10}
plt.rc('font', **font)   # pass in the font dict as kwargs
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

class Vertiport:
    
    def __init__(self, id = None, x = None, y = None, z = None):

        self.id     = id
        self.x      = x
        self.y      = y
        self.z      = z

class User:
    
    def __init__(self, id = None, Departure = None, Arrival = None):
    
        self.id             = id
        self.Arrival        = Arrival    # 도착 Vertiport의 no.
        self.Departure      = Departure  # 출발 Vertiport의 no.
        self.connect        = { 'drone'     : -1,
                                'isOnboard' : 0,
                                'isArrive'  : 0 }

    def is_support(self):

        return self.connect['drone'], self.connect['isOnboard'], self.connect['isArrive']

class Drone:
    
    def __init__(self, id = None, x = None, y = None, z = None, vertiport = None):
        
        self.id             = id
        self.x              = x
        self.y              = y
        self.z              = z
        self.vertiport      = vertiport
        self.onboarding     = 0
        self.counter        = 4
        self.takeoff        = 0
        self.MTOW           = MTOW
        self.numUser        = 0
        self.distance       = 0
        self.isworking      = 1
        self.seat           = np.full([self.MTOW], -1, dtype=int)
        self.batteryRemain  = 4 * 5.4 * 100000000 # [J]
        self.maxbattery     = self.batteryRemain
        self.num            = 0
        self.numidx         = np.zeros([N_VERTIPORT])
        self.numidx[vertiport] = 1

    def batteryConsump(self, action):
        
        if action == 4 or action == 6 or action == 7 or action == 8 or action == 9 or action == 10:
            batteryconsumption = 622 * 1000 * 60   # [J] Hovering
        elif action == 5:
            batteryconsumption = - self.maxbattery * 0.2 / 5
        else:
            batteryconsumption = 230 * 1000 * 60   # [J] Round-trip traveling

        self.batteryRemain -= batteryconsumption
        self.batteryRemain = min(self.batteryRemain, self.maxbattery)
        self.batteryRemain = max(self.batteryRemain, 0)
        if self.batteryRemain == 0: self.isworking = 0

    def _avail_action(self, userList, vertiportList):

        avail_action = np.ones(15)

        # Searching request action
        if self.x >=  GRID or self.z == H_MIN : avail_action[0] = 0
        if self.x <= -GRID or self.z == H_MIN : avail_action[1] = 0
        if self.y >=  GRID or self.z == H_MIN : avail_action[2] = 0
        if self.y <= -GRID or self.z == H_MIN : avail_action[3] = 0
        if self.x >= (GRID  / math.sqrt(2)) or self.y >= (GRID  / math.sqrt(2)) or self.z == H_MIN : avail_action[4] = 0
        if self.x <= (-GRID / math.sqrt(2)) or self.y >= (GRID  / math.sqrt(2)) or self.z == H_MIN : avail_action[5] = 0
        if self.x >= (GRID  / math.sqrt(2)) or self.y <= (-GRID / math.sqrt(2)) or self.z == H_MIN : avail_action[6] = 0
        if self.x <= (-GRID / math.sqrt(2)) or self.y <= (-GRID / math.sqrt(2)) or self.z == H_MIN : avail_action[7] = 0

        # Transportation action
        avail_action[8:] = 0 # 8 ~ 14
        
        if self.z == H_MAX:
            for v in range(N_VERTIPORT): # 10 11 12 13 14
                diff = math.sqrt((vertiportList[v].x - self.x)**2 + (vertiportList[v].y - self.y)**2)
                if diff <= VELOCITY: avail_action[v+10] = 1 # Vertiport에서 수직하강하는 action

            if self.vertiport != -1: avail_action[self.vertiport+10] = 0

        if self.z == H_MIN:
            if self.counter == 5:
                avail_action[8] = 1 # Vertiport에서 수직이륙하는 action
                avail_action[9] = 0 # waiting to charge at vertiport
            
            elif self.counter < 5:
                avail_action = np.zeros(15)
                avail_action[9] = 1 # waiting to charge at vertiport

        return avail_action

    def arrive_process(self, userList, idx, id, vertiportList):
        userList[id].connect['isOnboard'] = 0
        userList[id].connect['isArrive'] = 1
        # 해당 Passenger를 내렸으므로, UAM의 상태 Update
        self.seat[idx] = -1
        self.takeoff += 1

        x_diff = vertiportList[userList[id].Departure].x - vertiportList[userList[id].Arrival].x
        y_diff = vertiportList[userList[id].Departure].y - vertiportList[userList[id].Arrival].y
        self.distance = math.sqrt((x_diff)**2 + (y_diff)**2)

    def onboard_process(self, user, idx):
        user.connect['drone'] = self.id
        user.connect['isOnboard'] = 1
        self.seat[idx] = user.id

    def transition(self, action, userList, vertiportList):

        if self.isworking == 1:
            if   action == 0: self.x += VELOCITY
            elif action == 1: self.x -= VELOCITY
            elif action == 2: self.y += VELOCITY
            elif action == 3: self.y -= VELOCITY
            elif action == 4: 
                self.x += VELOCITY / math.sqrt(2)
                self.y += VELOCITY / math.sqrt(2)

            elif action == 5:
                self.x -= VELOCITY / math.sqrt(2)
                self.y += VELOCITY / math.sqrt(2)

            elif action == 6:
                self.x += VELOCITY / math.sqrt(2)
                self.y -= VELOCITY / math.sqrt(2)

            elif action == 7:
                self.x -= VELOCITY / math.sqrt(2)
                self.y -= VELOCITY / math.sqrt(2)

            elif action == 8: # Vertiport에서 수직이륙하는 action
                self.z = H_MAX
                self.counter = 0

            elif action == 9: # waiting to charge at vertiport
                self.counter += 1

                empty_mask = (self.seat == -1)
                empty_idx = [i for i, x in enumerate(empty_mask) if x ]
                for i in range(len(empty_idx)):
                    idx = empty_idx[i]
                    for user in userList:
                        if (user.connect['isArrive'] == 0) and (user.connect['isOnboard'] == 0) and (user.Departure == self.vertiport):
                            self.onboard_process(user, idx)
                            break

            elif action >= 10: # 수직착륙하는 action
                self.vertiport = action - 10
                self.numidx[self.vertiport] += 1
                self.x, self.y, self.z = vertiportList[self.vertiport].x, vertiportList[self.vertiport].y, H_MIN

                seat_mask = (self.seat != -1)
                seat_idx = [i for i, x in enumerate(seat_mask) if x]

                # 내릴 Vertiport 결정된 이후의 Transition
                for i in range(len(seat_idx)):  # 탑승한 Passenger 존재
                    idx = seat_idx[i]           # 탑승한 좌석 위치
                    id = self.seat[idx]         # 탑승한 Passenger id
                    if (userList[id].Arrival == vertiportList[self.vertiport].id):
                        self.arrive_process(userList, idx, id, vertiportList) # 해당 Passenger를 내렸으므로, UAM의 상태 Update
                        
                empty_mask = (self.seat == -1)
                empty_idx = [i for i, x in enumerate(empty_mask) if x]

                for i in range(len(empty_idx)):
                    idx = empty_idx[i]

                    for user in userList:
                        if (user.connect['isArrive'] == 0) and (user.connect['isOnboard'] == 0) and (user.Departure == self.vertiport):
                            self.onboard_process(user, idx)
                            break

        # state space 맞추어주기.
        self.x = self.clamp(self.x, -GRID, GRID)
        self.y = self.clamp(self.y, -GRID, GRID)
        self.batteryConsump(action)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

class Utility:
    
    def __init__(self):
        self.T_service_rate     = 0
        self.T_onboard_rate     = 0
        self.T_service_user     = 0
        self.T_onboard_user     = 0
        self.grid               = 60
        self.scale              = self.grid / GRID
        self.I_reward           = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_battery          = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_onboard_user     = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_service_user     = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.I_onboard_distance = np.zeros(N_COMMAGENT+N_DQNAGENT)
        self.SERVICED           = np.zeros(N_COMMAGENT+N_DQNAGENT+1)
        self.num_Takeoff        = np.zeros([N_VERTIPORT])

    def get_utils_info(self):

        return self.T_service_rate, self.I_battery

    def _calculate_support(self, userList, agentList, vertiportList):
        
        for i in range(N_USER):
            droneId, isOnboard, isArrive = userList[i].is_support()
            # print('droneID {} isOnboard {} isArrive {}'.format(droneId, isOnboard, isArrive))
            self.T_service_user += isArrive
            self.T_onboard_user += isOnboard
            if droneId != -1:
                self.I_service_user[droneId] += isArrive
                self.I_onboard_user[droneId] += isOnboard

        for j, agent in enumerate(agentList):
            seat_mask = (agent.seat != -1)
            seat_idx  = [a for a, b in enumerate(seat_mask) if b]

            total_distance = 0
            for i in range(len(seat_idx)): # 탑승한 Passenger 존재
                idx = seat_idx[i] # 탑승한 좌석 위치
                id = agent.seat[idx] # 탑승한 Passenger id

                x_diff          = agent.x - vertiportList[userList[id].Arrival].x
                y_diff          = agent.y - vertiportList[userList[id].Arrival].y
                total_distance += math.sqrt((x_diff)**2 + (y_diff)**2)
            
            if len(seat_idx) == 0:  self.I_onboard_distance[j] = total_distance / (GRID)
            else:                   self.I_onboard_distance[j] = total_distance / ( len(seat_idx) * GRID )
 
    def _calculate_energy_consumption(self, agentList):

        for drone in agentList:
            self.I_battery[drone.id]  = drone.batteryRemain

    def _calculate_indiv_reward(self, agentList, vertiportList):

        for i, drone in enumerate (agentList):
            takeoff_mask = (agentList[i].numidx > 0)
            takeoff_idx  = [m for m, n in enumerate(takeoff_mask) if n]
            # self.I_reward[i] = (agentList[i].takeoff + len(takeoff_idx) / len(vertiportList) + self.I_battery[i] / drone.maxbattery + drone.distance / GRID)
            # self.I_reward[i] = (agentList[i].takeoff + agentList[i].numidx.sum() / len(vertiportList) - self.I_onboard_distance[i] + self.I_battery[i] / drone.maxbattery)
            # self.I_reward[i] = (agentList[i].takeoff + len(takeoff_idx) - self.I_onboard_distance[i] + self.I_battery[i] / drone.maxbattery + drone.distance / GRID)
            # self.I_reward[i] = ((agentList[i].takeoff + len(takeoff_idx) - self.I_onboard_distance[i] + drone.distance / GRID)) * (self.I_battery[i] > 0)
            # self.I_reward[i] = ((agentList[i].takeoff + len(takeoff_idx) - self.I_onboard_distance[i])) * (self.I_battery[i] > 0)

            # 23.2.23 부터 다시 시작하는 Reward Shaping #
            self.I_reward[i] = (agentList[i].takeoff + len(takeoff_idx) - self.I_onboard_distance[i] + self.I_battery[i] / drone.maxbattery) # type 1
            # self.I_reward[i] = (agentList[i].takeoff + len(takeoff_idx) / len(vertiportList) - self.I_onboard_distance[i] + self.I_battery[i] / drone.maxbattery) # type 2

            # print(f'Indiv : {agentList[i].takeoff} | {len(takeoff_idx)} | {self.I_onboard_distance[i]} | {self.I_battery[i] / drone.maxbattery} | {drone.distance /GRID}')

        return self.I_reward

    def _calculate_common_reward(self, userList, agentList, vertiportList):

        for i, agent in enumerate(agentList): # self.num_Takeoff = np.zeros([self.numAgent, N_VERTIPORT])
            self.num_Takeoff += agent.numidx

        takeoff_mask  = (self.num_Takeoff > 0)
        takeoff_idx   = [i for i, x in enumerate(takeoff_mask) if x]

        # Common_Reward = np.full((N_COMMAGENT+N_DQNAGENT), self.T_service_user / (N_DQNAGENT + N_COMMAGENT)) * np.full((N_COMMAGENT+N_DQNAGENT), min(self.num_Takeoff)*2) # Comm
        Common_Reward = np.full((N_COMMAGENT+N_DQNAGENT), self.T_service_user / (N_DQNAGENT + N_COMMAGENT)) # * np.full((N_COMMAGENT+N_DQNAGENT), min(self.num_Takeoff)*2)
        # print(f'Comm : {self.T_service_user / (N_DQNAGENT + N_COMMAGENT)} | {np.mean(self.num_Takeoff)}')

        return Common_Reward
            
    def calculate_reward(self, userList, agentList, vertiportList):
        
        self.__init__() # 초기화 먼저 해줌
        self._calculate_support(userList, agentList, vertiportList)
        self._calculate_energy_consumption(agentList)
        self.SERVICED[0]                            = self.T_service_user
        self.SERVICED[1:(N_COMMAGENT+N_DQNAGENT+1)] = np.copy(self.I_service_user)
        Common_Reward                               = self._calculate_common_reward(userList, agentList, vertiportList)
        Individual_Reward                           = self._calculate_indiv_reward(agentList, vertiportList)
        # print(f'Indiv : {Individual_Reward}, Comm : {Common_Reward}')
        Reward                                      = np.array(Individual_Reward) + Common_Reward
        
        return Reward / 100.0
    
class Environment:
 
    def __init__(self):
        
        # [1] Environment 
        self.EPLEN        = EP_LEN
        self.numVertiport = N_VERTIPORT
        self.numUser      = N_USER
        self.numCommAgent = N_COMMAGENT
        self.numDQNAgent  = N_DQNAGENT
        self.numAgent     = self.numCommAgent + self.numDQNAgent
        self.GridRadius   = GRID
        self.hMax         = H_MAX
        self.hMin         = H_MIN
        self.t            = 0
        self.numTakeoff   = np.zeros([self.numAgent, N_VERTIPORT])

        # [2] Environment --> RLAgent 
        self.o            = None
        self.o_prime      = None
        
        # [3] Environment Initialization
        self.Initialize()
        
    def reset(self):
        self.Initialize()
    
    def get_available_action(self):
        available_action = []
        
        for i in range(self.numAgent):
            available_action.append(self.agentList[i]._avail_action(self.userList, self.vertiportList))
        return np.array(available_action)
    
    def Initialize(self):
        vertiportList = self._initVertiport()
        self._initUser()
        self._initAgent(vertiportList)

        self.utility = Utility()
        self.common_obs = self.getCommonObs()
        # self.common_obs_prime = np.copy(self.common_obs)
        self.o = self.getagentObs(self.t)
        self.o_prime = np.copy(self.o)
        self.ava_prime = self.get_available_action()
        
    def get_info(self):
        info = dict()
        info['episode_limit'] = self.EPLEN
        info['n_Comm_agents'] = self.numCommAgent
        info['n_DQN_agents']  = self.numDQNAgent
        info['n_agents']      = self.numCommAgent + self.numDQNAgent
        info['n_actions']     = 15
        info['obs_dim']       = self.o.shape[-1]
        return info


    def _initVertiport(self):
        self.vertiportList = []
        for i in range(self.numVertiport):
            if  i == 0:
                x, y = 0, 0 # [m]
                self.vertiportList.append(Vertiport(id=i, x=x, y=y, z=H_MIN))
            elif i == 1:
                x, y = -26192 , -17445
                self.vertiportList.append(Vertiport(id=i, x=x, y=y, z=H_MIN))
            elif i == 2:
                x, y = 19937, 22918
                self.vertiportList.append(Vertiport(id=i, x=x, y=y, z=H_MIN))
            elif i == 3:
                x, y = 21794, -13731
                self.vertiportList.append(Vertiport(id=i, x=x, y=y, z=H_MIN))
            elif i == 4:
                x, y = 17396, -7183
                self.vertiportList.append(Vertiport(id=i, x=x, y=y, z=H_MIN))

        return self.vertiportList

    def _initUser(self):
        self.userList = []
        for i in range(self.numUser):
            Departure = np.random.randint(0, self.numVertiport) # 1, 2, 3, 4번 Vertiport
            Arrival = np.random.randint(0, self.numVertiport)

            while(Departure == Arrival):
                Departure = np.random.randint(0, self.numVertiport)
                Arrival = np.random.randint(0, self.numVertiport)
            self.userList.append(User(id=i, Departure=Departure, Arrival=Arrival))

    def _initAgent(self, vertiportList):
        self.agentList = []
        marker = np.zeros([self.numVertiport], dtype=int)
        for i in range(self.numAgent):
            idx = np.random.randint(0, self.numVertiport)
            while marker[idx] > 1: # 0, 1
                idx = np.random.randint(0, self.numVertiport)
            marker[idx] += 1
            self.agentList.append(Drone(id=i, x=vertiportList[idx].x, y=vertiportList[idx].y, z=vertiportList[idx].z, vertiport=vertiportList[idx].id))

    # def _initAgent(self, vertiportList):
    #     self.agentList = []
    #     for i, vertiport in enumerate(vertiportList):
    #         self.agentList.append(Drone(id=2*i, x=vertiport.x, y=vertiport.y, z=vertiport.z, vertiport=vertiport.id))
    #         if i != 4: self.agentList.append(Drone(id=2*i+1, x=vertiport.x, y=vertiport.y, z=vertiport.z, vertiport=vertiport.id))


    def plot(self):
        fig, ax = plt.subplots(figsize=(4,4),dpi=150)
        DX, DY, DR = [],[],[]
        AX, AY, AR = [],[],[]
        CX, CY, CR = [],[],[]
        OX, OY = [],[]

        for i in range(self.numVertiport):
            DX.append(self.vertiportList[i].x)
            DY.append(self.vertiportList[i].y)

        for i in range(N_COMMAGENT):
            CX.append(self.agentList[i].x)
            CY.append(self.agentList[i].y)
            
        for i in range(N_COMMAGENT, self.numAgent):
            AX.append(self.agentList[i].x)
            AY.append(self.agentList[i].y)

        if len(DX):
            ax.scatter(DX, DY, s  = 90, c = 'k', marker = 'o')
            # for i in range(N_VERTIPORT):
            #     a_circle = plt.Circle((DX[i], DY[i]), DR[i], fill=False, color='yellowgreen', linewidth=2.5)
            #     ax.add_artist(a_circle)

        if len(CX):
            ax.scatter(CX, CY, s  = 90, c = 'magenta', marker = '*')
            # for i in range(N_COMMAGENT):
            #     a_circle = plt.Circle((CX[i], CY[i]), CR[i], fill=False, color='magenta', linewidth=2.5)         
            #     ax.add_artist(a_circle)
                
        if len(AX):
            ax.scatter(AX, AY, s  = 90, c = 'dodgerblue', marker = '*')
            # for i in range(self.numAgent-N_COMMAGENT):
            #     a_circle = plt.Circle((AX[i], AY[i]), AR[i], fill=False, color='dodgerblue', linewidth=2.5)
            #     ax.add_artist(a_circle)

        if len(OX):
            ax.scatter(OX, OY, s  = 90, c = 'orangered', marker = 'X')

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.xlim([-self.GridRadius, self.GridRadius])
        plt.ylim([-self.GridRadius, self.GridRadius])
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        plt.close()
        image = ToTensor()(image).unsqueeze(0)[0]
        return image
    
    def getCommonObs(self):
    
        # [1] 요소에 대한 절대 위치

        # agent_pos = np.array([[self.agentList[k].x, self.agentList[k].y] for k in range(self.numAgent)]).flatten()
        vertiport_pos = np.array([[self.vertiportList[k].x, self.vertiportList[k].y] for k in range(self.numVertiport)]).flatten()
        # user_pos = np.array([[self.vertiportList[self.userList[k].Departure].x, self.vertiportList[self.userList[k].Departure].y] for k in range(self.numUser)]).flatten()
        # Position = np.hstack([agent_pos,vertiport_pos,user_pos])
        Position = np.hstack([vertiport_pos])
        ## Normalize
        Position = Position / self.GridRadius
        
        # # [2] Connection 정보
        # userisConnect = np.array([[user.request, user.connect['drone'] / (self.numAgent), user.connect['isOnboard'], user.connect['isArrive']] \
        #                           for user in self.userList]).flatten()
        # Connection    = np.hstack([userisConnect])
        
        # [3] Utility 정보
        Util          = np.array([self.utility.T_service_rate, self.utility.T_onboard_rate])
        Util          = np.append(Util, np.array([self.utility.I_service_user, self.utility.I_onboard_user, self.utility.I_onboard_distance]).flatten())
        
        # 모든 Observation을 concatenate
        # CommonObs     = np.hstack([Position, Connection, Util])
        CommonObs = np.hstack([Position, Util])
        return CommonObs

    def getagentObs(self, t):
        
        Obs = []
        obs_common = self.common_obs

        for i in range(self.numAgent):
            # o_partial = []

        # [1] 자신의 정보

            # agent = self.agentList[i] # 현재 주체가 되는 Agent의 index == i
            o_pos = []
            o_pos.append(self.agentList[i].x / self.GridRadius)
            o_pos.append(self.agentList[i].y / self.GridRadius)
            o_pos.append(self.agentList[i].z / self.GridRadius)
            o_pos.append(self.agentList[i].batteryRemain / self.agentList[i].maxbattery)
            # o_pos.append(self.agentList[i].batteryRemain)

        # [2] 다른 Agent의 위치 정보

            o_pos_agent = []
            for k in range(self.numAgent):
                x_diff = self.agentList[i].x - self.agentList[k].x
                y_diff = self.agentList[i].y - self.agentList[k].y
                diff   = math.sqrt((x_diff)**2 + (y_diff)**2)

                if diff <= OBSERVABLE:
                    o_pos_agent.append(self.agentList[k].x / self.GridRadius)
                    o_pos_agent.append(self.agentList[k].y / self.GridRadius)
                    # o_pos_agent.append(self.agentList[k].z / self.GridRadius)
                    o_pos_agent.append(diff / self.GridRadius)

                else: 
                    o_pos_agent.append(-1)
                    o_pos_agent.append(-1)
                    # o_pos_agent.append(-1)
                    o_pos_agent.append(-1)

        # [3] Onloaded Passenger의 정보

            o_seat = []

            seat_mask = (self.agentList[i].seat != -1)
            seat_idx = [j for j, x in enumerate(seat_mask) if x]

            seat_counter = 0
            for k in range(MTOW):
                if seat_mask[k]:
                    idx = seat_idx[seat_counter]
                    seat_counter += 1
                    id = self.agentList[i].seat[idx]

                    # o_seat.append(self.userList[id].connect['drone'] / self.numAgent)
                    # o_seat.append(self.userList[id].connect['isOnboard'])
                    # o_seat.append(self.userList[id].connect['isArrive'])
                    o_seat.append(self.userList[id].Arrival / N_VERTIPORT)
                    o_seat.append(self.userList[id].Departure / N_VERTIPORT)

                else:
                    # o_seat.append(-1)
                    o_seat.append(-1)
                    o_seat.append(-1)

        # [4] Vertiport의 위치 정보

            o_pos_vertiport = []
            for k in range(self.numVertiport):
                x_diff = self.agentList[i].x - self.vertiportList[k].x
                y_diff = self.agentList[i].y - self.vertiportList[k].y
                diff   = math.sqrt((x_diff)**2 + (y_diff)**2)

                if diff <= OBSERVABLE:
                    o_pos_vertiport.append(self.vertiportList[k].x / self.GridRadius)
                    o_pos_vertiport.append(self.vertiportList[k].y / self.GridRadius)
                    # o_pos_vertiport.append(self.vertiportList[k].z / self.GridRadius)
                    o_pos_vertiport.append(diff / self.GridRadius)

                else: 
                    o_pos_vertiport.append(-1)
                    o_pos_vertiport.append(-1)
                    # o_pos_vertiport.append(-1)
                    o_pos_vertiport.append(-1)

            o_pos           = np.array(o_pos)
            o_pos_agent     = np.array(o_pos_agent)
            o_pos_vertiport = np.array(o_pos_vertiport)
            o_seat          = np.array(o_seat)

            o_idx = np.zeros(self.numAgent)
            o_idx[self.agentList[i].id] += 1
            o_partial = np.hstack([o_pos, o_pos_agent, o_pos_vertiport, o_seat]) # 모든 정보 Concatenation
            # o = np.hstack([o_idx, obs_common, o_partial])
            # o = np.hstack([t/EP_LEN, o_idx, o_partial])
            o = np.hstack([o_idx, o_partial])
            Obs.append(np.copy(o))
        Obs = np.array(Obs)
        return Obs


    def step(self, actions, time, epoch, today, info):
        self.t = time
        # print(actions)
        self.o = np.copy(self.o_prime)
        for i in range(self.numAgent):
            self.agentList[i].transition(actions[i], self.userList, self.vertiportList)
        # self.getCommonObs()
        self.o_prime = self.getagentObs(time)
        
        rewards = self.utility.calculate_reward(self.userList, self.agentList, self.vertiportList)

        if(time == 0):
            self.x_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.y_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.z_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.x_DNN  = np.zeros([EP_LEN, N_DQNAGENT])
            self.y_DNN  = np.zeros([EP_LEN, N_DQNAGENT])
            self.z_DNN  = np.zeros([EP_LEN, N_DQNAGENT])

        if(time == EP_LEN):
            for i, agent in enumerate(self.agentList):
                self.numTakeoff[i] = agent.numidx
            
        self.ava_prime = self.get_available_action()

        if (time < EP_LEN):
            for i in range(N_COMMAGENT):
                self.x_COMM[time, i] = self.agentList[i].x
                self.y_COMM[time, i] = self.agentList[i].y
                self.z_COMM[time, i] = self.agentList[i].z

            for i in range(N_COMMAGENT, self.numAgent):
                self.x_DNN[time, i-N_COMMAGENT] = self.agentList[i].x
                self.y_DNN[time, i-N_COMMAGENT] = self.agentList[i].y
                self.z_DNN[time, i-N_COMMAGENT] = self.agentList[i].z

        return self.o, rewards, self.o_prime, self.ava_prime, self.numTakeoff, self.x_COMM, self.y_COMM, self.x_DNN, self.y_DNN