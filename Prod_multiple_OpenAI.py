import random
import copy
import time

import numpy as np
from matplotlib import pyplot as plt

from NASH import T, n, ptypes, p_sequence, parts, mp, n_wait, n_SB, D, s_criticalIndx, B, b, Tp, MTTR, ng, MTBF, gs, ms, \
    Tl, Tu, t, null_part, omega_d, omega_s


class Multiproduct:

    def __init__(self, ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu):
        # self.m = m
        self.ptypes = ptypes  # product_types
        self.n = n  # n_machines
        self.b = b  # buffer_level
        self.B = B  # buffer_capacity
        self.Tp = Tp  # process_time
        self.Tl = Tl  # load_time
        self.Tu = Tu  # unload_time
        self.ng = ng  # n_gantries
        self.MTTR = MTTR
        self.MTBF = MTBF
        self.ms = ms  # machine_state
        self.gs = gs  # gantry_state
        self.T = T  # simulation time
        self.t=0
        # self.machines = machines
        self.n_SB = n_SB
        self.n_wait = n_wait
        self.processing = [False, False, False, False]  # machine busy or not
        self.p_sequence = p_sequence  # sequence of each part through the line
        self.mp = mp
        self.parts = parts
        self.prod_count = np.zeros([n, ptypes])
        self.prev_prod_count = np.zeros([n, ptypes]) # for reward calculation
        self.cum_parts= np.zeros(n)  # cumulative parts produced by each machine
        self.TBF = copy.deepcopy(MTBF)
        self.TTR = copy.deepcopy(MTTR)
        self.status=True  # just for code checkup, it s also machine status
        self.W= np.zeros(n) # machine downtime, code checkup
        self.terminated = False
        self.Tr= np.zeros(n)
        self.waiting_time = np.zeros(n)
        self.prod_count = np.zeros([n, ptypes])
        self.mready=[False, False, False, False]   # if ready to process or not
        self.mprogress = np.zeros(n)  # processed parts on each machine irrespective of type
        self.downtime= np.zeros(n)
        self.total_downtime =  np.zeros(n)
        self.g_load= np.zeros(n) # which machine is being loaded
        self.g_unload= np.zeros(n)  # which machine is being unloaded
        self.loading= np.zeros(n)
        self.unloading = np.zeros(n)
        self.unload_check= [False, False, False, False]
        # self.agents= ['Robot_1', 'Robot_2']
        self.agents = [0,1]

    def get_alist(self):  # action list
        # self.actionList = []
        # for i in range(ptypes):
        #     for j in range(n):
        #         if p_sequence[i][j] == 1:
        #             self.actionList.append((i, j))
        # # action_selected= random.choice(self.actionList)
        # return self.actionList

        return [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]

    def run_machine(self, j):  # j is the machine number
        if np.any(self.mp[j] > np.zeros(ptypes)): # if machine has a part
            if self.ms[j] == 1: # if machine is running or down
                self.downtime[j] = 0
                if self.mready[j]==True:  # processing not started yet, part loaded,
                    if self.Tr[j] == 0:
                        self.processing[j] = True  # start part processing, initiating the part processing
                        self.mprogress[j]+= 1/Tp[j]
                        self.Tr[j] += 1
                    else:
                        self.mprogress[j] += 1 / Tp[j]
                        self.Tr[j] += 1
                        if self.mprogress[j]>=1 or self.Tr[j]==self.Tp[j]:
                            self.mprogress[j]=0
                            self.Tr[j]=0
                            self.cum_parts[j]+=1
                            self.processing[j]= False
                            self.n_wait[j]=1
                            self.mready[j]=False
                            self.waiting_time[j]+= 1
                else:
                    if self.n_wait[j]==0 and self.processing[j]==True:
                        self.loading[j] += 1
                        self.g_load[j]=1
                        if self.loading[j] == self.Tl[j]:
                            self.loading[j] = 0
                            self.gs[j] = 0
                            self.g_load[j] = 0
                            self.mready[j]= True
            else:
                self.downtime[j]+=1
                self.total_downtime += 1
        else:
            self.n_wait[j] = 1  # machine j is waiting to be loaded
            self.processing[j] = False
            self.waiting_time[j]+=1
            self.mready[j]= False
        return self.Tr[j], self.cum_parts[j], self.waiting_time[j], self.downtime[j], self.n_wait[j]

    def load_machine(self,j):
        if self.n_wait[j]==1:
            if np.sum(self.mp[j])==0:
                if j!=0:
                    if np.sum(self.b[j - 1]) > 0:
                        # print("np.sum(self.b[j - 1]): ", np.sum(self.b[j - 1]))
                        # print("machine: ", j)
                        # print("self.n_wait[j]: ", self.n_wait[j])
                        # print("self.mp[j]: ", self.mp[j])
                        self.gs[j]=1
                        self.g_load[j]=1
                    else:
                        self.n_SB[j]=1
                        # print('previous buffer is empty')
                else:
                    self.gs[j] = 1
                    self.g_load[j] = 1
        #     else:
        #         print('machine has already a part')
        # else:
        #     print('machine needs to be in waiting')

    def unload_machine(self, j):
        if self.n_wait[j] == 1:
            if np.sum(self.mp[j]) > 0:
                if j!=(n-1):
                    if self.next_buffer_has_space(j)==True:
                    # if np.sum(self.b[j])< self.B[j]:
                        self.gs[j]=1
                    else:
                        self.n_SB[j]=1
                        # print('next buffer is full')
                else:
                    self.gs[j] = 1
        #     else:
        #         print('machine is already empty')
        # else:
        #     print('machine needs to be in waiting')

    def run_gantry(self, action):
        part, machine = action
        if (np.sum(self.gs)< self.ng) and self.gs[machine] == 0:  # if gantry is available and machine is not assigned a gantry
            if self.ms[machine]==1:  # if machine to be loaded is up
                if np.sum(self.mp[machine])==0:    # machine has no part and is waiting to be loaded
                    if not self.processing[machine] and not self.mready[machine]:
                        if machine!=0 and np.sum(b[machine-1])>0: #(self.n_SB[machine]==0 or np.sum(b[machine-1])
                            self.load_machine(machine)
                            self.n_wait[machine]=0
                            self.run_buffer(action)
                            self.processing[machine] = True
                            self.Tr[machine]=0
                            # self.mp[machine][part] = self.parts[part][part]
                        if machine==0:
                            self.load_machine(machine)
                            self.n_wait[machine] = 0
                            self.run_buffer(action)
                            self.processing[machine] = True
                            self.Tr[machine] = 0
                            self.mp[machine][part] = self.parts[part][part]
                    # else:
                    #     print('check if there is any part in machine and whether it is SB')
                elif not self.processing[machine] and not self.mready[machine] and self.n_wait[machine]==1 and self.Tr[machine]==0 and (((machine != (n - 1) and np.sum(self.b[machine])< self.B[machine] and (np.sum(self.b[machine-1])> 0) or machine==0))):
                        # print("machine to be unloaded: ", machine)
                        # print("self.mp[machine]", self.mp[machine])
                        # print("self.n_SB[machine]", self.n_SB[machine])
                        # # print("self.b[machine]", self.b[machine])
                        # print("self.b[machine-1]", self.b[machine-1])
                        self.unload_machine(machine)
                        self.plus_buffer(action)
                        self.mp[machine]= [0,0,0]
                        self.unload_check[machine]= True
                        # if self.unloading[machine] == self.Tu[machine]:
                        # if machine==(n-1):
                        #     self.prod_count[machine][part] += parts[part, part]
                        self.load_machine(machine)
                        self.n_wait[machine] = 0
                        self.processing[machine]= True
                        self.waiting_time[machine]+= 1
                        if machine != 0:
                            self.minus_buffer(action)
                        else:
                            self.mp[machine][part] = 1
                elif machine==(n-1) and np.sum(self.b[machine-1])>0:
                    self.unload_machine(machine)
                    self.plus_buffer(action)
                    self.mp[machine] = [0, 0, 0]
                    self.unload_check[machine] = True
                    # if self.unloading[machine] == self.Tu[machine]:
                    # if machine==(n-1):
                    #     self.prod_count[machine][part] += parts[part, part]
                    self.load_machine(machine)
                    self.n_wait[machine] = 0
                    self.processing[machine] = True
                    self.waiting_time[machine] += 1
                    if machine != 0:
                        self.minus_buffer(action)
                    else:
                        self.mp[machine][part] = 1

        elif (self.gs[machine] == 1) and not self.processing[machine] and not self.mready[machine] and self.n_wait[machine]==1 and self.Tr[machine]==0 and np.sum(self.mp[machine])==0:
            self.load_machine(machine)
            if self.n_SB[machine] != 1:
                self.n_wait[machine] = 0
                self.processing[machine]= True
                if machine!=0:
                    self.minus_buffer(action)
                else:
                # if np.all(self.b[machine - 1] >= self.parts[part]):
                    self.mp[machine][part] = 1
        elif np.sum(gs) == ng:
            self.n_wait[machine]=1
            self.processing[machine] = False
            self.mready[machine] = False

        return self.gs, self.mp

    def minus_buffer(self, action):
        part, machine = action
        if np.all(self.mp[machine] == np.zeros(self.ptypes)):
            if np.all(self.b[machine - 1] >= self.parts[part]):
                # self.mp[machine, part] = 1
                self.b[machine - 1][part] -= 1
                self.mp[machine][part] = 1
            else:
                idxs = []
                sp = []
                for idx, val in enumerate(self.b[machine - 1]):
                    if val != 0:    #[0,0,0]
                        idxs.append(idx)
                        sp.append(val)
                if len(sp)>0:
                    selected_part= self.b[machine - 1].index(sp[0])
                    self.mp[machine][selected_part] = 1
                    self.b[machine - 1][selected_part] -= 1
                # else:
                #     self.n_SB[machine] = 1

    def plus_buffer(self, action):
        part, machine = action

        if np.any(self.mp[machine] > np.zeros(self.ptypes)) and machine!=(n-1):
            part_to_unload = list(self.mp[machine]).index(1)  # it turns out the index of the part already on the machine. The index also represents the part type
            part_to_unload_sequence = list(self.p_sequence[part_to_unload])  # let say [1,0,1,1]
            # for k in part_to_unload_sequence:  # maybe we can use np.where() function after this step
            if part_to_unload_sequence[machine + 1] == 1:  # maybe a loop can be used to check the next process of the part in sequence
                if self.B[machine] > np.sum(self.b[machine]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1  # new action: assign part j to machine i
                # else:
                #     self.n_SB[machine] = 1
            elif part_to_unload_sequence[machine + 2] == 1:
                if self.B[machine + 1] > np.sum(self.b[machine + 1]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine + 1][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1
                # else:
                #     self.n_SB[machine] = 1
            elif part_to_unload_sequence[machine + 3] == 1:
                if self.B[machine + 2] > np.sum(self.b[machine + 2]):  # checking which next machine has an operation on this unloaded part
                    self.b[machine + 2][part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    self.prod_count[machine, part_to_unload] += self.parts[part_to_unload, part_to_unload]
                    # self.mp[machine, part] = 1
                # else:
                #     self.n_SB[machine] = 1
        if machine == (n-1) and np.any(self.mp[machine] > np.zeros(self.ptypes)):
            part_to_unload = list(self.mp[machine]).index(1)
            self.prod_count[machine][part_to_unload]+=1

    def next_buffer_has_space(self, machine):
        if np.sum(self.mp[machine]) > 0 and machine!=(n-1):
            part_to_unload = list(self.mp[machine]).index(1)
            part_to_unload_sequence = list(self.p_sequence[part_to_unload])

            if part_to_unload_sequence[machine + 1] == 1:
                if self.B[machine] > np.sum(self.b[machine]):
                    return True
                else:
                    return False
            elif machine != 2 and part_to_unload_sequence[machine + 2] == 1:
                if self.B[machine + 1] > np.sum(self.b[machine + 1]):
                    return True
                else:
                    return False
            elif machine == 0 and part_to_unload_sequence[machine + 3] == 1:
                if self.B[machine + 2] > np.sum(self.b[machine + 2]):
                    return True
                else:
                    return False
            else:
                return False
        if machine == (n-1) and np.sum(self.mp[machine]) > 0:
            return True

    def run_buffer(self, action):  # action: (part, mach)
        part, machine = action
        # gantry_status= self.Gantry(action)
        # if np.sum(gantry_status) < ng:
        if machine == 0:  # first machine
            self.plus_buffer(action)
        if machine != 0 and machine!=(n-1):  # In-between machines
            self.plus_buffer(action)
            self.minus_buffer(action)
        if machine==(self.n-1):    #last machine
            self.minus_buffer(action)

        return self.b, self.mp, self.prod_count  # we may take the parts with machine from the machine but i think it should be fine here as well...

    def reset(self):  # should we reset the buffers to zero and prod.count to zero
        self.Tr = [0] * self.n
        self.processing = [0] * self.n  # it is either True or False for each machine i.e. processing or not
        self.gs = [0] * self.n
        self.mp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        self.n_SB= [0] * self.n
        self.n_wait= [1] * self.n
        self.t=0
        self.ms = [1] * self.n
        # self.b= np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]]).tolist()
        b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).tolist()
        return (1, 1, 1, 1,0,0,0,0, 0,0,0,0, 0, 0, 0, 0, 0, 0, 0)
        # return (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
        # return (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0)

    def get_state(self):
        s = [self.ms[j] for j in range(n)] #+ [self.mp[j] for j in range(n)]
        # # Add buffer status to the state
        # s += [self.b[j] for j in range(len(self.b))]  # self.b is already a list of lists
        s+= [sum(self.mp[j]) for j in range(len(self.mp))] # part with each machine

        # Adding mprogress to the state, just for openAI control
        s+= [self.mprogress[j] for j in range(n)]

        # Add normalized buffer status (sum and normalize each sublist)
        max_capacity = 15 #max([sum(b) for b in self.b])  # Define max capacity based on your buffer setup
        # s += [sum(self.b[j]) / max_capacity for j in range(len(self.b))]  # Normalized buffer sums
        s += [sum(self.b[j]) for j in range(len(self.b))]
        # Add gantry state
        s += [self.gs[j] for j in range(len(self.gs))]
        a = tuple(s)
        mstates = tuple(y for x in a for y in (x if isinstance(x, list) else (x,)))
        # print("state_updated: ", mstates)
        return mstates

    def get_prod_count(self):
        self.m0_prod= []
        self.m1_prod = []
        self.m2_prod = []
        self.m3_prod = []
        self.m0_prod.append(self.prod_count[0])
        self.m1_prod.append(self.prod_count[1])
        self.m2_prod.append(self.prod_count[2])
        self.m3_prod.append(self.prod_count[3])
        return self.m0_prod, self.m1_prod, self.m2_prod, self.m3_prod

    def get_reward(self):
        # cod = np.zeros(ptypes)
        # cos = np.zeros(ptypes)
        # # if self.ms[s_criticalIndx]==0:   # slowest critical machine index # reward needs to be changed to D/Tp[slowest_critical]
        # ppl= self.downtime[s_criticalIndx]/(self.Tl[s_criticalIndx] + self.Tp[s_criticalIndx] + self.Tu[s_criticalIndx])
        # reward=0
        # for j in range(ptypes):
        #     if D[j]> self.prod_count[-1][j]:
        #         cod[j]= D[j]-self.prod_count[-1][j]
        #     else:
        #         cod[j]=0
        # CoD= np.sum(omega_d * cod)
        #
        # for j in range(ptypes):
        #     if self.prod_count[-1][j]> D[j]:
        #         cos[j]= 0 #self.prod_count[-1][j]- D[j]
        #     else:
        #         cos[j]=0
        # CoS= np.sum(omega_d * cos)
        # CoT = CoD + CoS
        #
        # reward= -ppl-CoT
        # reward = np.sum(self.prod_count[-1]) * 1
        # print("reward: ", reward)
        # waiting_penalty=0
        # for machine_id, status in enumerate(self.n_wait):
        #     if status == 0:  # Machine is idle/waiting
        #         waiting_penalty -= 0.1  # Penalize machine for waiting
        step_production = np.sum(self.prod_count[-1]) - np.sum(self.prev_prod_count[-1])

        # Update the previous production count for the next step
        self.prev_prod_count = self.prod_count.copy()

        # Return the reward for the current step
        reward = step_production * 1  # Modify the scaling factor as needed
        total_reward= reward #+waiting_penalty
        return total_reward

    def get_info(self):
        return {"Prod_counts": self.prod_count[self.n - 1], "Gantry_state": self.gs, 'Machine_state': self.ms,
                'Parts_with_machine': self.mp}

    def get_W(self):
        for j in range(self.n):
            if self.W[j] == 0:
                if self.TBF[j] == 0:
                    self.W[j] = 1
                    self.ms[j] = 0   # down
                    self.TBF[j] = np.random.geometric(1 / self.MTBF[j])
                else:
                    self.TBF[j] -= 1
            if self.W[j] == 1:
                if self.TTR[j] == 0:
                    self.W[j] = 0
                    self.ms[j] = 1  # up
                    self.TTR[j] = np.random.geometric(1 / self.MTTR[j])
                else:
                    self.TTR[j] -= 1

    def mp_has_nullPart(self):
        # self.mp = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
        null_part = [0, 0, 0]
        result = []
        for i in self.mp:
            if np.all(i == null_part):
                result.append(1)
            else:
                result.append(0)
        if np.sum(result) == 0:
            test = False
        else:
            test = True
        return test

    def select_action(self,machine_index):
        feasible_actions = []
        for i in self.actionList:
            if i[1] == machine_index:
                feasible_actions.append(i)
        selected_action = random.choice(feasible_actions)
        # print(feasible_actions)
        return selected_action

    def machine_to_be_unloaded(self):
        machine_to_be_unload = None
        indxs_mprogress = []
        indxs_Tr = []
        for i, j in enumerate(self.mprogress):
            if j == 0:
                indxs_mprogress.append(i)
        # print(indxs_mprogress)
        for l, m in enumerate(self.Tr):
            if m == 0:
                indxs_Tr.append(l)
        # print(indxs_Tr)
        for i in indxs_mprogress:
            for j in indxs_Tr:
                # print(i,j)
                if i == j and self.processing[i] == False and self.mready[i] == False:
                    machine_to_be_unload = i
                # else:
                #     print('No machine is ready')
        if machine_to_be_unload != None:
            return machine_to_be_unload

    def next_machine_to_be_unloaded(self):
        next_machine_to_be_unload= None
        for i in range(3):   # len(b)= 3
            if i < 2:
                if self.next_buffer_has_space(i+1) == True:
                    if np.sum(self.b[i]) > 0 and self.processing[i + 1] == False and self.mready[i + 1] == False and self.mprogress[i + 1] == 0:
                        next_machine_to_be_unload = (i + 1)
                        break
                    else:
                        if self.next_buffer_has_space(0) == True:
                            if self.processing[0] == False and self.mready[0] == False and self.mprogress[0] == 0:
                                next_machine_to_be_unload = 0
                                break
                else:
                    if self.next_buffer_has_space(0) == True:
                        if self.processing[0] == False and self.mready[0] == False and self.mprogress[0] == 0:
                            next_machine_to_be_unload = 0
                            break
            elif np.sum(b[i]) > 0 and self.processing[i + 1] == False and self.mready[i + 1] == False and self.mprogress[i + 1] == 0:
                    next_machine_to_be_unload = (i + 1)
                    break
            # else:
            #     print('no machine is ready, SB')
        if next_machine_to_be_unload != None:
            return next_machine_to_be_unload
    def get_done(self):
        return False

    def get_obs_space(self):
        obs= self.get_state()
        return np.array(obs)

    def get_action_space(self):
        actions= self.get_alist()
        return np.array(actions)

    def step(self, actions):
        # Check if actions are in the correct format
        try:
            # Ensure actions is a list of exactly two tuples
            if not (isinstance(actions, list) and len(actions) == 2 and
                    all(isinstance(action, tuple) and len(action) == 2 for action in actions)):
                raise ValueError("Actions must be a list of two tuples.")

            # Proceed with the normal operation
            [self.run_machine(k) for k in range(n)]
            actionList = self.get_alist()

            for action in actions:  # actions should be like [(0, 1), (1, 2)]
                if any(np.array_equal(sublist, [0, 0, 0]) for sublist in mp):
                    part, machine = action
                    if np.all(self.mp[machine] == [0, 0, 0]):
                        self.get_W()
                        if self.W[machine] == 0:
                            self.run_gantry(action)
                            self.run_machine(machine)
                        else:
                            self.downtime[machine] += 1
                            self.total_downtime[machine] += 1
                    else:
                        if (self.mprogress[machine] == 0 and self.Tr[machine] == 0 and
                                not self.processing[machine] and self.n_wait[machine] == 1 and
                                not self.mready[machine]):
                            self.run_gantry(action)
                            self.run_machine(machine)
                else:
                    if (np.any(self.mprogress == 0) and np.any(self.Tr == 0) and
                            np.any(self.processing == False) and np.any(self.n_wait == 1) and
                            np.any(self.mready == False)):
                        part, machine = action
                        self.run_gantry(action)
                        self.run_machine(machine)

            return self.get_state(), self.get_reward(), self.get_done(), self.get_info()

        except ValueError as e:
            print(f"Error: {e}")
            # Optionally return a default value or raise an error
            return None, None, None, None  # Or handle accordingly

    # def step(self, actions):
    #     # part, machine = action
    #     # print("action in the step: ", actions)
    #     [self.run_machine(k) for k in range(n)]
    #     actionList= self.get_alist()
    #     # action_pairs = [actionList[i] for i in actions]
    #     # print("action final in the step: ", action_pairs)
    #     for action in actions: # the action coming is like this [[(0,1), (1,2)]], that's why taking inner list only
    #         if any(np.array_equal(sublist, [0, 0, 0]) for sublist in mp):
    #             part, machine = action
    #             # print("part and machine in step: ", part, machine)
    #             if np.all(self.mp[machine]== [0,0,0]):
    #                 self.get_W()
    #                 # print('self.W[machine]=', self.W[machine])
    #                 if self.W[machine]==0:
    #                     self.run_gantry(action)
    #                     self.run_machine(machine)
    #                     # self.run_buffer(action)
    #                 else:
    #                     self.downtime[machine]+=1
    #                     self.total_downtime[machine]+=1
    #             else:
    #                 if self.mprogress[machine] == 0 and self.Tr[machine] == 0 and self.processing[machine] == False and self.n_wait[machine] == 1 and self.mready[machine] == False:
    #                     self.run_gantry(action)
    #                     self.run_machine(machine)
    #         else:
    #             if np.any(self.mprogress == 0) and np.any(self.Tr == 0) and np.any(self.processing == False) and \
    #                     np.any(self.n_wait == 1) and np.any(self.mready == False):
    #                 part, machine = action
    #                 self.run_gantry(action)
    #                 self.run_machine(machine)
    #                 # self.get_reward()
    #     # print("self.b: ", self.b)
    #     return self.get_state(), self.get_reward(), self.get_done(), self.get_info()

from openai import OpenAI
import ast

# Initialize OpenAI client




def get_chatgpt_action(state):
    # Define the prompt based on the state #     The goal is to maximize production throughput, specifically focusing on the number of parts processed by the last machine (Machine 4). When selecting
    #     actions, prioritize loading downstream machines (machine 4 first, then machine 3, and so on).

    prompt = f"""You must only provide the actions without any details for example '[(0,2), (0,3)]'.
    Consider a serial manufacturing system with four machines and three in-between buffers. Each machine has a dedicated processing time, 
    and the buffers have maximum capacities of 15 parts. There are two robots in the system to handle materials.

    Given the following states:

    Machine 1 running status: {state[0]}
    Machine 2 running status: {state[1]}
    Machine 3 running status: {state[2]}
    Machine 4 running status: {state[3]}
    Machine 1 mp-status: {state[4]}
    Machine 2 mp-status: {state[5]}
    Machine 3 mp-status: {state[6]}
    Machine 4 mp-status: {state[7]}
    Buffer 1 level: {state[12]}
    Buffer 2 level: {state[13]}
    Buffer 3 level: {state[14]}
    Machine 1 progress: {state[8]}
    Machine 2 progress: {state[9]}
    Machine 3 progress: {state[10]}
    Machine 4 progress: {state[11]}
    Machine 1 robot status: {state[15]}
    Machine 2 robot status: {state[16]}
    Machine 3 robot status: {state[17]}
    Machine 4 robot status: {state[18]}

    Evaluate the following actions:

    Action 1 (0,0): Load machine 1
    Action 2 (0,1): Load machine 2
    Action 3 (0,2): Load machine 3
    Action 4 (0,3): Load machine 4

    The feasibility criteria for each action are as follows:

    - Action 1 is feasible only if Buffer 1 < 15, Machine 1 is running, Machine 1 mp-status is free or processing with no progress, and Machine 1 robot status is free.
    - Action 2 is feasible if Buffer 1 > 0, Buffer 2 < 15, Machine 2 is running, Machine 2 mp-status is free or processing with no progress, and Machine 2 robot status is free.
    - Action 3 is feasible if Buffer 2 > 0, Buffer 3 < 15, Machine 3 is running, Machine 3 mp-status is free or processing with no progress, and Machine 3 robot status is free.
    - Action 4 is feasible if Buffer 3 > 0, Machine 4 is running, Machine 4 mp-status is free or processing with no progress, and Machine 4 robot status is free.

    If only one action is feasible, it must be accompanied by Action 1. This means if, for example, only Action 3 is feasible, the output should look like this: [(0,2), (0,0)]

    If multiple actions are feasible, the actions must be prioritized as follows:

    Prioritization of Actions:
    1. Action 4 (0,3) must be prioritized the most.
    2. Action 3 (0,2) should be prioritized next.
    3. Action 2 (0,1) should be prioritized less than Action 3.
    4. Action 1 (0,0) must be prioritized the least.

    If no actions are feasible, return '[(0,0), (0,0)]' to indicate both robots are idle.

    Based on these states, criteria, and the goal of maximizing production throughput by prioritizing downstream machines, what actions should be assigned to the two robots? The output must be in the format '[(action1), (action2)]' without additional details.
    """

    #The output must only provide the actions with no details for example [(0,1), (1,3)].
    # Call OpenAI API with the new client
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-4",  # or "gpt-4", gpt-3.5-turbo
        max_tokens=10,  # Adjust as needed
        temperature=0.3
    )
    # print("response from gpt: ", response)
    print("prompt: ", prompt)
    action_content = response.choices[0].message.content.strip()
    print('action:', action_content)  # Output: "[(0,0), (1,0)]"

    try:
        action_list = ast.literal_eval(action_content)
        print('action list:', action_list)  # Print the resulting list of tuples
    except SyntaxError as e:
        print(f"SyntaxError: {e}. Action content was: '{action_content}'")
        action_list = [(0, 0), (0, 0)]  # Fallback to default action
    except Exception as e:
        print(f"Error: {e}. Action content was: '{action_content}'")
        action_list = [(0, 0), (0, 0)]  # Fallback to default action
    return action_list

def evaluate_chatgpt_agent(env, steps_per_episode=100, runs_per_episode=1):
    log_data = []
    rewards_for_runs = []

    last_action = None  # Initialize last_action outside the loop to track across steps

    # Run the single episode for 3 runs
    for run in range(runs_per_episode):
        state = env.reset()
        # print("state from reset: ", state)
        run_reward = 0

        for step in range(steps_per_episode):
            print("Step: ", step)
            # print("state: ", state)
            # print(f"Before GPT action: Machine 1 running status: {state[0]},Machine 2 running status: {state[1]}, Machine 3 running status: {state[2]},Machine 4 "
            #       f"running status: {state[3]} Machine 1 mp-status: {state[4]} Machine 2 mp-status: {state[5]}Machine 3 mp-status: {state[6]}"
            #       f"Machine 4 mp-status: {state[7]} Buffer 1 level: {state[8]} Buffer 2 level: {state[9]} Buffer 3 level: {state[10]} "
            #       f"Machine 1 progress:{state[11]} Machine 2 progress:{state[12]} Machine 3 progress:{state[13]} Machine 4 progress: {state[14]}"
            #       f"Machine 1 robot status: {state[15]} Machine 2 robot status: {state[16]} Machine 3 robot status: {state[17]} Machine 4 robot "
            #       f"status: {state[18]}")
            print("State provided: ", state)
            action = get_chatgpt_action(state)

            # Take the step in the environment using the action
            next_state, reward, done, info = env.step(action)
            run_reward += reward
            state = next_state
            # print("state from step ftn: ", state)
            # Update last_action with the current action
            # last_action = action
            print("run reward: ", run_reward)
            # Introduce a small delay to avoid hitting API rate limits
            time.sleep(1)

        rewards_for_runs.append(run_reward)
        print(f"Run {run + 1}: Total Reward: {run_reward}")

    # Calculate mean reward and standard deviation across the runs
    mean_reward = np.mean(rewards_for_runs)
    std_reward = np.std(rewards_for_runs)

    return mean_reward, std_reward

def plot_rewards(mean_reward, std_reward):
    # Plot the reward as a bar with error bars (std deviation)
    plt.bar(0, mean_reward, yerr=std_reward, alpha=0.7, capsize=5, color='blue')
    plt.xticks([0], ['Episode 1'])
    plt.xlabel('Episode')
    plt.ylabel('Mean Total Reward')
    plt.title('Mean Reward for One Episode with Std Deviation (GPT-4)')
    plt.show()

if __name__ == "__main__":

    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)

    # Evaluate the ChatGPT agent for one episode run 3 times
    mean_reward, std_reward = evaluate_chatgpt_agent(env, steps_per_episode=50, runs_per_episode=1)

    # Plot the rewards
    plot_rewards(mean_reward, std_reward)

