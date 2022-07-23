import gym
import numpy as np
from gym import error, spaces
from transfer import transfer
import time
import sys
import os
if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
    #which means in my diannao
    sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA542-gym/KrigingPython')
else :
    # which means in 106 server
    sys.path.append(r'C:/Users/106/Desktop/DDPGshishi/DDPG-master/CDA542-gym/KrigingPython')
from Surrogate_01de import Surrugate

class CDA542Env(gym.Env):
    metadata = {'render.modes': ['human']}

    if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
        #which means in my diannao
        script_folder = 'C:/Users/y/Desktop/temp/testCDA1'
        matlab_location = 'C:/Users/y/Desktop/temp/MXairfoilCDA'
    else:
        # which means in 106 server
        script_folder = 'C:/Users/106/Desktop/temp/testCDA1'
        matlab_location = 'C:/Users/106/Desktop/temp/MXairfoilCDA'        

    def __init__(self,**kargs):
        print('MXairfoil: CDA542Env Environment initialized. En Taro XXH!')
        # trying to make it to be commonly used.
        self.viewer=None
        self.server_process=None
        self.server_port = None
        
        if 'real_dim' in kargs:
            self.real_dim = kargs['real_dim'] # useless in fact. initial function is not explicitely called.
        else:
            self.real_dim = 4
        # self.dx = 0.1 # this is for 'virtual grid'
        # self.dx = 0 # raw state must be changed when changing self.dx.
        self.dx = 0.1 # this dx defines the extension, for the input of surrogate mode. by using this, parameters in [-1,1] would change, while ones in [0,1] would not.

        self.transfer = transfer(dim=self.real_dim,dx=self.dx)
        self.constraints_real = np.array([1.0513,35.8])
        self.constraints_dim = len(self.constraints_real)
        self.state = np.zeros((self.real_dim+self.constraints_dim,),dtype=np.float64)# [chi_in, chi_out] # fu** you 'int 32'
        self.set_constraints(self.constraints_real)

        if 'constraints_apply' in kargs:
            self.constraints_apply = kargs['constraints_apply']
        else:
            self.constraints_apply = False

        self.action_space = spaces.Box(low=np.ones((self.real_dim,))*(-1.0), high=np.ones((self.real_dim,))*(1.0), shape=(self.real_dim, ), dtype=np.float64)

        self.real_action_space = 0.025 # this can be set, but dynamic step size sounds lack of evidence
        self.real_action_space_h = np.ones((self.real_dim,),dtype=np.float64)*self.real_action_space
        self.real_action_space_l = np.ones((self.real_dim,),dtype=np.float64)*self.real_action_space*(-1.0)

        #this is for Sorrogate model
        self.diaoyong = Surrugate()
        self.diaoyong.load()

        # 4 action dims and two constraints, and a reward.
        self.observation_space = spaces.Box(low=np.ones((self.real_dim+self.constraints_dim,))*(-1.0), high=np.ones((self.real_dim+self.constraints_dim,))*(1.0), shape=(self.real_dim+self.constraints_dim, ), dtype=np.float64)

        
        self.real_obs_space_h = np.ones((self.real_dim,),dtype=np.float64)+self.dx # constraints do not use dx.
        self.real_obs_space_l = np.zeros((self.real_dim,),dtype=np.float64)-self.dx
        # the so-called "real" here, is the input of surrogate model

        self.normal_action_space_h =np.ones((self.real_dim,),dtype=np.float64)
        self.normal_action_space_l =np.ones((self.real_dim,),dtype=np.float64)*(-1.0)
        self.normal_obs_space_h =np.ones((self.real_dim+self.constraints_dim,),dtype=np.float64)
        self.normal_obs_space_l =np.ones((self.real_dim+self.constraints_dim,),dtype=np.float64)*(-1.0)
        
        # self.performance = np.array([0.0,0.0,0.0])
        self.performance = np.array([0.06,1.05,30.0,0.5])
        #[omega, rise , turn,reward], since state are used for constraints, a new member of class CDAenv** is needed to transfer the performance.

        # this is for reset_random
        self.random_r = 0.08 #0.08
        self.decay = 0.8

        # this is for 'artificial tip'
        # self.omega_target = 0.06596

        self.N_step = 0
        self.N_artificial_tip = 0

        self.reward=0.0
        # self.raw_state = np.array([ 0.0,0.0]) # [chi_in, chi_out]. it would change when using high-order feedback. 
        self.raw_state = np.zeros((self.real_dim,),dtype=np.float64)
        self.supplement_2D = np.array([0.4675,0.41045233]) # this should in surrogate space [0,1], not normal space [-1,1]
        # normal state = [ 0.04078333 -0.03896875 -0.05416667 -0.14924611]
        # surrogate state = [0.52447    0.47661875 0.4675     0.41045233]

        # self.constraints_normal = self.transfer.real_to_normal_constraints(np.array([1.0513,35.8]))

        
        self.get_reward_1()
        
        # attention, defination of self.state are changed here, for dynamic constraints.

    def real_to_norm_action(self,action):
        #change action from real to normal.
        real_action_bili = ( self.real_action_space_h - self.real_action_space_l ) /2
        real_action_c = ( self.real_action_space_h + self.real_action_space_l ) /2
        norm_action = (action - real_action_c) / real_action_bili
        return norm_action

    def norm_to_real_action(self,action):
        #change action from real to normal.
        real_action_bili = ( self.real_action_space_h - self.real_action_space_l ) /2
        real_action_c = ( self.real_action_space_h + self.real_action_space_l ) /2
        real_action = action*real_action_bili + real_action_c
        return real_action

    def real_to_norm_state(self,state):
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2
        real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        norm_state = (state - real_state_c) / real_state_bili
        return norm_state

    def norm_to_real_state(self,state):
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2
        real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        real_state = state*real_state_bili + real_state_c
        return real_state


    def jisuan5(self):
        # this is to add some restrict into our model. so-called dynamic model for generalization ability
        # virtual grid, artificial tip, high order feedback are used.
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
            # print('artificial tip used: ' + str(self.N_artificial_tip))

        X_surrogate = self.state[0:self.real_dim]*1.0

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state * 0.0
        # dx_surrogate = np.zeros((self.real_dim+2,))
        # this will always positive or zero. It records how much the state exceed the [0-1]

        for i in range(self.real_dim):
            if(X_surrogate[i]>1):
                dx_surrogate[i]=X_surrogate[i] - 1
                X_surrogate[i] = 1
            elif(X_surrogate[i]<0):
                dx_surrogate[i]=0 - X_surrogate[i]
                X_surrogate[i] = 0
        penalty_x = dx_surrogate.sum() * 3 * 10 * self.dx # this constants represents the strength of penalty.
        if self.real_dim == 2:
            X_surrogate_in = np.append(X_surrogate,self.supplement_2D)
        else:
            X_surrogate_in = X_surrogate
        
        omega = self.diaoyong.k1.predict(X_surrogate_in)
        rise = self.diaoyong.k2.predict(X_surrogate_in)
        turn = self.diaoyong.k3.predict(X_surrogate_in)

        # # dynamic constraints are applied here, not yet, zhaoba.
        # if self.state[5]>25:
        #     # which means constraints is in real.
        #     constraints_real = self.state[4:6]
        #     print('MXairfoil: there must be something wrong in CDA step')
        # else:
        #     constraints_real = self.transfer.normal_to_real_constraints(self.state[4:6])

        
        if self.constraints_apply:
            restrict_rise = self.constraints_real[0]
        else:
            restrict_rise = -114514 # uncomment this to disable the constraints. 
        if rise<restrict_rise :
            penalty_rise = -0.2 - (restrict_rise - rise) * 500
        else:
            penalty_rise=0

        if self.constraints_apply:
            restrict_turn = self.constraints_real[1]
        else:
            restrict_turn = -114514 # uncomment this to disable the constraints. 
        if turn<restrict_turn : # the bigger the better.
            penalty_turn = -0.2 - (restrict_turn - turn) * 0.05
        else:
            penalty_turn=0

        self.reward = -500*omega + 28.2 - penalty_x + penalty_rise + penalty_turn

        try:
            self.reward = self.reward[0]
            # reward should be float but sometimes it can be np.array, so a fault tolerance is here.
        except:
            pass
        

        # these lines is for so-called 'artificial tip'.
        deltar_art = 0.08

        # # this is first kind of artificial tip
        # if omega < self.omega_target:
        #     # which means it is good enough, so it should be rewarded more.
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.


        raw_state_surrogate = self.norm_to_real_state(self.raw_state[0:self.real_dim])
        jvli = np.sum((self.state[0:self.real_dim]-raw_state_surrogate)**2)
        deltax = 0.0009 # 0.0025
        if self.reward>0.65:
            # which means agent (raw_state in it, in in fact) is relatively better.
            # but might be changed if
            # this is another kind of artificial tip.
            if jvli < deltax:
                # # which means this point is close enough.
                self.reward = self.reward + deltar_art
                self.N_artificial_tip = self.N_artificial_tip+1
                print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))
                # print('MXairfoil:artificial tip disabled.')

        # # this is first kind of artificial tip. expired. tuofu is not bai learn.
        # omega_target = 0.05515
        # if jvli < deltax*16 and self.raw_state[-1]>0.6 and omega < omega_target:
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art
        #     self.N_artificial_tip = self.N_artificial_tip+1
        #     print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))
        self.performance[0] = omega
        self.performance[1] = rise
        self.performance[2] = turn
        self.performance[3] = self.reward

    def step(self,x):
        self.N_step=self.N_step+1
        x = np.array(x)
        try:
            x = x.reshape(self.real_dim,)
        except:
            print('MXairfoil: unmatched dimensions, G! ')
            os.system('pause')

        x = self.norm_to_real_action(x)

        assert self.action_space.contains(x) , 'MXairfoil error: invalid step input'
        real_dim = self.real_dim

        self.state[0:real_dim] = self.norm_to_real_state(self.state[0:real_dim])
        self.state[0:real_dim] = self.state[0:real_dim] + x

        notdone = 1
        for i in range(real_dim):
            if(self.state[i]>self.real_obs_space_h[i]):
                self.state[i]=self.real_obs_space_h[i]-0.00001
                notdone = 0
            elif(self.state[i]<self.real_obs_space_l[i]):
                self.state[i]=self.real_obs_space_l[i]+0.00001
                notdone = 0
        # call surrogate model and calculate.
        # self.jisuan2() # original
        # self.jisuan3() # 'enough different'
        # self.jisuan4() # 'artificial tip'
        self.jisuan5() # restrict added into it.
        # self.jisuan6() # totally new architecture, trying to get more generalization ability by using dynamic constraints.

        self.state[0:real_dim] = self.real_to_norm_state(self.state[0:real_dim])
        # done2 = 0
        # if(self.state[4]>self.normal_obs_space_h[4]):
        #     # fail because omega is too high
        #     done2 = 1
        # if(self.state[5]<self.normal_obs_space_l[5]):
        #     # fail because rise is too
        #     done2 = 1

        # self.reward=self.state[-1]
        # reward was calculated in jisuan2, to be more centralized

        # notdone = np.isfinite(self.state).all()
        notdone = np.isfinite(self.state).all() & notdone

        done = not notdone
        # done = done | self.diaoyong.done # if fail to calculate, get out of this thread.
        # done = done | done2 #if out of range, get out of this thread. # it looks unnecessary

        return self.state, self.reward, done, {}

    def reset(self):
        # print('ShishiEnv Environment reset')
        self.reward=0.0
        # self.state = np.array([-0.9511,-0.0187,-0.1883,0.2925,0.0,0.0,0.0]).reshape(7,) #[x,y]
        # self.state = np.random.uniform(-1,1,(7,))
        self.state = np.zeros((self.real_dim+self.constraints_dim,),dtype=np.float64)
        self.set_constraints_random()
        self.get_reward_1()
        # self.N_step = 0
        return self.state

    def render(self):
        strbuffer = 'MXairfoil: x='+str(self.state[0:self.state.size-4])+'  omega ='+str(self.state[self.state.size-3]) + '  rise ='+str(self.state[self.state.size-2]) + '  reward ='+str(self.state[self.state.size-1])
        self.diaoyong.jilu(strbuffer)
        print('En Taro XXH!')

    def reset_random(self):
        # reset to a random state. this is for trainning
        print('CDA542 Environment reset into limited random state, training')
        self.reward=0.0
        self.state = self.raw_state*1.0
        self.set_constraints_random()

        # this is another kind. A r is guaranteed, to strenthen exploration.
        # this is much more complex when real_dim = 4 ; # these lines are universal. I m so congming!.
        theta = np.random.uniform(0,2*np.pi,(self.real_dim-1,))
        r = self.random_r
        dstate = np.zeros((self.real_dim,),dtype=np.float64)
        dstate[0] = r 
        for i in range(self.real_dim-1): # basic operation, sit down and no 666 please.
            dstate[i] = dstate[i]*np.cos(theta[i])
            dstate[i+1] = dstate[i]*np.sin(theta[i])
        # check = dstate**2
        # check = check.sum()
        self.state[0:self.real_dim] = self.state[0:self.real_dim] + dstate.reshape(self.real_dim,)
        self.N_step = 0 
        self.N_for_decay = 0 
        # self.state[5] = self.N_for_decay*1 # reuse the 'rise' dimension.
        return self.state

    def reset_original(self):
        # reset to original state. this is for testing the generalization ability of my reinforcement learning model
        print('CDA542 Environment reset into original state')
        self.reward=0.0
        original_state = np.array([ 0.04078333 -0.03896875 -0.05416667 -0.14924611])
        self.state[0:self.real_dim] = original_state[0:self.real_dim]
        # normal state = [ 0.04078333 -0.03896875 -0.05416667 -0.14924611]
        # surrogate state = [0.52447    0.47661875 0.4675     0.41045233]
        self.set_constraints_random()
        self.get_reward_1()
        # self.N_step = 0
        return self.state

    def reset_random2(self):
        # reset to a random state. this is for testing the generalization ability of my reinforcement model
        print('CDA542 Environment reset into totally random state.')
        # random_normal_constraints = np.random.uniform(-0.9,0.9,(2,))
        # random_real_constraints = self.transfer.normal_to_real_constraints(random_normal_constraints)

        self.state = self.raw_state
        self.set_constraints_random()
        self.state[0:self.real_dim] = np.random.uniform(-0.9,0.9,(self.real_dim,))
        self.get_reward_1()
        return self.state

    def set_raw_state(self,new_raw_state):
        print('MXairfoil: raw state updated when N_step = '+str(self.N_step)+' \nbefore: '+str(self.raw_state) + '\nafter: '+ str(new_raw_state) +'\nand decay the randomness')
        # normal constraints are included in raw_state and raw_state_new.

        if len(self.raw_state)!= len(new_raw_state):
            print('MXairfoil: unmatched dimension in raw state, G!') 
            os.system('pause')
        # self.raw_state[0:self.real_dim] = new_raw_state[0:self.real_dim]
        self.raw_state = new_raw_state

        # decay the randomness when settig raw_state.
        self.random_r = self.random_r * self.decay

    def set_state(self,state):
        # this is for GA, directly set the state.
        # self.state = state
        state2 = np.array(state)
        chicun = state2.shape
        self.state[0:chicun[0]] = state2
        action0 = np.zeros((self.real_dim,))
        self.state, self.reward, done, asd = self.step(action0)
        self.N_step = self.N_step+1
        return self.state, self.reward, done, asd

    def set_constraints(self,real_constraints):
        # this is for dynamic constraints.
        # self.state[4] = real_constraints[0]
        # self.state[5] = real_constraints[1]
        print('MXairfoil: constraints in the env are set to be '+str(real_constraints))
        normal_constraints = self.transfer.real_to_normal_constraints(real_constraints)
        
        self.state[self.real_dim:self.real_dim+self.constraints_dim] = normal_constraints
        return self.state

    def set_constraints_random(self):
        # get a unified method to set the constraints into random.
        random_normal_constraints = np.random.uniform(-0.9,0.9,(2,))
        # random_normal_constraints = np.zeros((2,))
        self.state[self.real_dim:self.real_dim+self.constraints_dim] = random_normal_constraints
        # print('MXairfoil: constraints in the env are set to be random.')
        print('MXairfoil: constraints ... random.')
        return self.state
        # print('MXairfoil: constraints in the env are set to be statical so far.')

    def get_performance(self):
        # just get performace from here.
        # since self.performance is not private, this is not necessary in fact.
        return self.performance

    def get_reward_1(self):
        # this is to get reward for initaled state or reseted state, do not use 0.5 again.
        # real_dim = self.real_dim
        # self.state = self.norm_to_real_state(self.state)
        # self.jisuan5() # distinguishing step function and jisuan function, I'm very congming!
        # self.state = self.real_to_norm_state(self.state)

        self.step(np.zeros(self.real_dim,))# this is more yangjian than copy code.

    def extra_set(self,**kargs):
        # this is to synchronize the raw_state, performance and so on in agent and in environment.
        if 'raw_state_performance' in kargs:
            self.performance = kargs['raw_state_performance'].reshape(4,)*1.0
        if 'raw_state_save' in kargs:
            self.raw_state = kargs['raw_state_save'].reshape(self.real_dim+self.constraints_dim,)*1.0
        if 'constraints_real' in kargs:
            self.constraints_real = kargs['constraints_real']*1.0


class ShishiEnvExtend(gym.Env):
    def __init__(self):
        print('ShishiEnvExtend Environment initialized')
    def step(self):
        print('ShishiEnvExtend Step successful!')
    def reset(self):
        print('ShishiEnvExtend Environment reset')

