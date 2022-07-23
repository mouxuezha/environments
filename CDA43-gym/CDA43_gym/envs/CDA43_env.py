import gym
import numpy as np 
from gym import error, spaces
import math
from call_components import call_components
import time 
import sys
sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA43-gym/KrigingPython')
from Surrogate_01de import Surrugate

class CDA43Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    script_folder = 'C:/Users/y/Desktop/temp/testCDA1'
    matlab_location = 'C:/Users/y/Desktop/temp/MXairfoilCDA'


    def __init__(self):
        print('MXairfoil: CDA43Env Environment initialized. En Taro XXH!')
        self.viewer=None
        self.server_process=None
        self.server_port = None
        self.action_space = spaces.Box(low=-1.0, high= 1.0, shape=(2, ), dtype=np.float64)

        self.real_action_space = 0.025 # this can be set, but dynamic step size sounds lack of evidence
        self.real_action_space_h = np.array([self.real_action_space,self.real_action_space])
        self.real_action_space_l = np.array([-1*self.real_action_space,-1*self.real_action_space])
         
        self.observation_space = spaces.Box(low=-1.0, high= 1.0, shape=(7, ), dtype=np.float64)
        # self.dx = 0.1 # this is for 'virtual grid'
        # self.dx = 0 # raw state must be changed when changing self.dx.
        self.dx = 0.1 # this dx defines the extension, for the input of surrogate mode. by using this, parameters in [-1,1] would change, while ones in [0,1] would not.
        self.real_obs_space_h = np.array([1+self.dx,1+self.dx,1+self.dx,1+self.dx])
        self.real_obs_space_l = np.array([0-self.dx,0-self.dx,0-self.dx,0-self.dx])
        # the so-called "real" here, is the input of surrogate model

        self.normal_action_space_h =np.array([1,1])
        self.normal_action_space_l =np.array([-1,-1]) 
        self.normal_obs_space_h =np.array([1,1,1,1,1,1,1])
        self.normal_obs_space_l =np.array([-1,-1,-1,-1,-1,-1,-1]) 



        self.reward=0.0
        self.raw_state = np.array([ 0.04078333,-0.03896875,-0.05416667,-0.14924611,0.0,0.0,0.0])  
        # self.raw_state = np.array([-0.80205413,0.00315371,-0.05416667,-0.14924611,0.05512563,1.05150588,0.63718419])
        self.state = np.array([0,0,self.raw_state[2],self.raw_state[3],0.0,0.0,0.0]).reshape(7,)
        # [chi_in, chi_out, mxthk, umxthk, omega, rise, user-defined reward]
        # calculated by shishienv83-3.real_to_norm_state

        # this is for reset_random
        self.random_r = 0.16 #0.08 
        self.decay = 0.8

        # this is for 'artificial tip'
        # self.omega_target = 0.06596 

        #this is for Sorrogate model
        self.diaoyong2 = Surrugate()
        self.diaoyong2.load()

        self.N_step = 0 
        self.real_dim = 2
        self.N_artificial_tip = 0 

        self.performance_dim = 7 
        self.performance = np.zeros(self.performance_dim,dtype=float)        

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


    def jisuan2(self):
        #this is for surrogate model
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
        
        X_surrogate = self.state[0:self.real_dim+2]*1 # now real_dim is 2, so another +2 is needed.

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state[0:self.real_dim+2] * 0 
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


        omega = self.diaoyong2.k1.predict(X_surrogate)
        rise = self.diaoyong2.k2.predict(X_surrogate)

        # then, these code are same as self.jisuan() .
        self.state[self.state.size-3] = omega
        self.state[self.state.size-2] = rise
        self.state[self.state.size-1] = -100*omega + 7.2 - penalty_x # this is to modify the reward.

        # self.state[self.state.size-1] = -100*self.state[self.real_dim] + 7.2
        # self.state[self.state.size-1] = 100*(rise-1.04)
    
    def jisuan3(self):
        #this is a modification of jisuan2, to encourage explore. 
        # creat a name for it : 'enough different' stratigication
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
        
        X_surrogate = self.state[0:self.real_dim+2]*1 # now real_dim is 2, so another +2 is needed.

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state[0:self.real_dim+2] * 0 
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


        omega = self.diaoyong2.k1.predict(X_surrogate)
        rise = self.diaoyong2.k2.predict(X_surrogate)

        # then, these code are same as self.jisuan() .
        self.state[self.state.size-3] = omega
        self.state[self.state.size-2] = rise
        zhongjie = -100*omega + 7.2 - penalty_x
        # self.state[self.state.size-1] = -100*omega + 7.2 - penalty_x # this is to modify the reward.
        if zhongjie > (self.state[self.state.size-1] + 0.005):
            self.state[self.state.size-1] = zhongjie

        # self.state[self.state.size-1] = -100*self.state[self.real_dim] + 7.2
        # self.state[self.state.size-1] = 100*(rise-1.04)

    def jisuan4(self):
        #this is for an newer stratigication, to avoid flat. give an extra reward when it is good enough.
        # creat a name: 'artificial tip'
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
        
        X_surrogate = self.state[0:self.real_dim+2]*1 # now real_dim is 2, so another +2 is needed.

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state[0:self.real_dim+2] * 0 
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

        omega = self.diaoyong2.k1.predict(X_surrogate)
        rise = self.diaoyong2.k2.predict(X_surrogate)

        # then, these code are same as self.jisuan() .
        self.state[self.state.size-3] = omega
        self.state[self.state.size-2] = rise
        self.state[self.state.size-1] = -100*omega + 7.2 - penalty_x # this is to modify the reward.

        # these lines is for so-called 'artificial tip'.
        deltar_art = 0.03

        # # this is first kind of artificial tip 
        # if omega < self.omega_target:
        #     # which means it is good enough, so it should be rewarded more.
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.
        
        # this is another kind of artificial tip.
        jvli = np.sum((self.state[0:2]-self.raw_state[0:2])**2)
        deltax = 0.00005
        if self.raw_state[-1]>0.6:
            # which means agent (raw_state in it, in in fact) is relatively better.
            if jvli < deltax:
                # which means this point is close enough.
                self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.

    def jisuan5(self):
        # this is to add some restrict into our model. virtual grid, artificial tip, high order feedback are used.
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
            print('artificial tip used: ' + str(self.N_artificial_tip))
        
        X_surrogate = self.state[0:self.real_dim+2]*1 # now real_dim is 2, so another +2 is needed.

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state[0:self.real_dim+2] * 0 
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

        omega = self.diaoyong2.k1.predict(X_surrogate)
        rise = self.diaoyong2.k2.predict(X_surrogate)
        turn = self.diaoyong2.k3.predict(X_surrogate)

        # try a 'hard' restrict first. Add a penalty when rise is too low.
        penalty_rise = 0 
        restrict_rise = 1.0513 # the bigger the better.
        if rise<restrict_rise : 
            penalty_rise = -0.2 - (restrict_rise - rise) * 500
            # 1 is too big, makes everything devastated
            # 0.1 is too small, fail to restrict
        
        penalty_turn = 0
        restrict_turn = 35.8
        if turn<restrict_turn : # the bigger the better.
            penalty_turn = -0.2 - (restrict_turn - turn) * 0.05
            # 1 is too big, makes everything devastated
            # 0.1 is too small, fail to restrict


        # then, these code are same as self.jisuan() .
        self.state[self.state.size-3] = omega
        self.state[self.state.size-2] = rise
        self.state[self.state.size-1] = -500*omega + 28.2 - penalty_x + penalty_rise + penalty_turn# this is to modify the reward.

        # these lines is for so-called 'artificial tip'.
        deltar_art = 0.03

        # # this is first kind of artificial tip 
        # if omega < self.omega_target:
        #     # which means it is good enough, so it should be rewarded more.
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.
        
        
        raw_state_surrogate = self.norm_to_real_state(self.raw_state[0:self.real_dim+2])
        jvli = np.sum((self.state[0:2]-raw_state_surrogate[0:2])**2)
        deltax = 0.0025
        if self.raw_state[-1]>0.6:
            # which means agent (raw_state in it, in in fact) is relatively better.
            # but might be changed if 
            # this is another kind of artificial tip.
            if jvli < deltax:
                # which means this point is close enough.
                self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.
                # self.state[-1] = self.state[-1] + deltar_art
                self.N_artificial_tip = self.N_artificial_tip+1
                print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))

        # # this is first kind of artificial tip 
        # omega_target = 0.05515
        # if jvli < deltax*16 and self.raw_state[-1]>0.6 and omega < omega_target:
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art
        #     self.N_artificial_tip = self.N_artificial_tip+1
        #     print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))
        
        # update performance
        self.performance[0] = omega
        self.performance[1] = rise
        self.performance[2] = self.state[self.state.size-1]
        self.performance[-1] = self.state[self.state.size-1]

    def jisuan6(self):
        # this is to do something closer to "design". En taro XXH! .
        if self.N_step%1000 == 0:
            print('MXairfoil: surrogate model is working.'+'\nN_step ='+str(self.N_step))
            print('artificial tip used: ' + str(self.N_artificial_tip))
        
        X_surrogate = self.state[0:self.real_dim+2]*1 # now real_dim is 2, so another +2 is needed.

        # if state is in [0,1], then nothing will happen. But if not, there must be some extra operation.
        dx_surrogate = self.state[0:self.real_dim+2] * 0 
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

        omega = self.diaoyong2.k1.predict(X_surrogate)
        rise = self.diaoyong2.k2.predict(X_surrogate)
        turn = self.diaoyong2.k3.predict(X_surrogate)

        # try a 'hard' restrict first. Add a penalty when rise is too low.
        penalty_rise = 0 
        restrict_rise = 1.0513 # the bigger the better.
        if rise<restrict_rise : 
            penalty_rise = -0.2 - (restrict_rise - rise) * 500
            # 1 is too big, makes everything devastated
            # 0.1 is too small, fail to restrict
        
        penalty_turn = 0
        restrict_turn = 35.8
        if turn<restrict_turn : # the bigger the better.
            penalty_turn = -0.2 - (restrict_turn - turn) * 0.05
            # 1 is too big, makes everything devastated
            # 0.1 is too small, fail to restrict


        # then, these code are same as self.jisuan() .
        self.state[self.state.size-3] = omega
        self.state[self.state.size-2] = rise
        self.state[self.state.size-1] = -500*omega + 28.2 - penalty_x + penalty_rise + penalty_turn# this is to modify the reward.

        # these lines is for so-called 'artificial tip'.
        deltar_art = 0.03

        # # this is first kind of artificial tip 
        # if omega < self.omega_target:
        #     # which means it is good enough, so it should be rewarded more.
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.
        
        
        raw_state_surrogate = self.norm_to_real_state(self.raw_state[0:self.real_dim+2])
        jvli = np.sum((self.state[0:2]-raw_state_surrogate[0:2])**2)
        deltax = 0.0025
        if self.raw_state[-1]>0.6:
            # which means agent (raw_state in it, in in fact) is relatively better.
            # but might be changed if 
            # this is another kind of artificial tip.
            if jvli < deltax:
                # which means this point is close enough.
                self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art # this constant has no theoritical basis.
                # self.state[-1] = self.state[-1] + deltar_art
                self.N_artificial_tip = self.N_artificial_tip+1
                print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))

        # # this is first kind of artificial tip 
        # omega_target = 0.05515
        # if jvli < deltax*16 and self.raw_state[-1]>0.6 and omega < omega_target:
        #     self.state[self.state.size-1] = self.state[self.state.size-1] + deltar_art
        #     self.N_artificial_tip = self.N_artificial_tip+1
        #     print('MXairfoil:artificial tip used.' + str(self.N_artificial_tip))



    def step(self,x):
        self.N_step=self.N_step+1
        x = np.array(x)
        x = x.reshape(self.state.size-5,)
        x = self.norm_to_real_action(x)
        
        assert self.action_space.contains(x) , 'MXairfoil error: invalid step input'
        real_dim = self.real_dim
        self.state[0:real_dim+2] = self.norm_to_real_state(self.state[0:real_dim+2])
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
        
        self.state[0:real_dim+2] = self.real_to_norm_state(self.state[0:real_dim+2])
        # done2 = 0 
        # if(self.state[4]>self.normal_obs_space_h[4]):
        #     # fail because omega is too high
        #     done2 = 1
        # if(self.state[5]<self.normal_obs_space_l[5]):
        #     # fail because rise is too 
        #     done2 = 1

        self.reward=self.state[self.state.size-1]
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
        self.state = np.array([0,0,self.raw_state[2],self.raw_state[3],0.0,0.0,0.0]).reshape(7,) #[x,y] , in normal([-1,1])
        self.N_step = 0 
        return self.state

    def render(self):
        strbuffer = 'MXairfoil: x='+str(self.state[0:self.state.size-4])+'  omega ='+str(self.state[self.state.size-3]) + '  rise ='+str(self.state[self.state.size-2]) + '  reward ='+str(self.state[self.state.size-1])
        self.diaoyong.jilu(strbuffer)
        print('En Taro XXH!')   

    def reset_random(self):
        # reset to a random state. this is for trainning
        print('CDA43 Environment reset into limited random state, training')
        self.reward=0.0
        # self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape(7,) #[x,y]
        # self.state = np.array([0,0,self.raw_state[2],self.raw_state[3],0.0,0.0,0.0]).reshape(7,) 
        self.state = np.array([self.raw_state[0],  self.raw_state[1],self.raw_state[2],self.raw_state[3],0.0,0.0,0.0]).reshape(7,)
        # array([-0.16666667,  0.83333333, -0.15694445,  0.24378567]) # when dx = 0.1
        # -0.2,1, when dx = 0 
        # dstate = np.random.uniform(-0.1,0.1,(2,)) # this is a primary kind of randomness
        # this is another kind. A r is guaranteed, to strenthen exploration.
        theta = np.random.uniform(0,2*np.pi,(1,))
        # r = 0.08
        r = self.random_r
        dstate = np.array([r*np.cos(theta), r*np.sin(theta)])
        self.state[0:2] = self.state[0:2] + dstate.reshape(2,)
        self.N_step = 0 
        return self.state
    
    def reset_original(self):
        # reset to original state. this is for testing the generalization ability of my reinforcement learning model
        print('CDA43 Environment reset into original state')
        self.reward=0.0
        self.state =np.array([ 0.04078333,-0.03896875,-0.05416667,-0.14924611,0.0,0.0,0.0])
        self.N_step = 0 
        return self.state

    def reset_random2(self):
        # reset to a random state. this is for testing the generalization ability of my reinforcement model
        print('CDA43 Environment reset into totally random state, testing model')
        self.state = np.array([self.raw_state[0],  self.raw_state[1],self.raw_state[2],self.raw_state[3],0.0,0.0,0.0]).reshape(7,)
        self.state[0:2] = np.random.uniform(-0.9,0.9,(2,))
        return self.state
   
    def set_raw_state(self,new_raw_state):
        print('MXairfoil: raw state updated when N_step = '+str(self.N_step)+' \nbefore: '+str(self.raw_state) + '\nafter: '+ str(new_raw_state) +'\nand decay the randomness')

        self.raw_state[0:self.real_dim] = new_raw_state[0:self.real_dim]
        # self.raw_state = new_raw_state

        # decay the randomness when settig raw_state.
        self.random_r = self.random_r * self.decay

    def set_state(self,state):
        # this is for GA, directly set the state.
        # self.state = state 
        state2 = np.array(state)
        chicun = state2.shape
        self.state[0:chicun[0]] = state2 
        action0 = np.array([0,0])
        self.state, self.reward, done, asd = self.step(action0)
        self.N_step = self.N_step+1
        return self.state, self.reward, done, asd
    
    def get_performance(self):
        # just get performace from here.
        return self.performance

class ShishiEnvExtend(gym.Env):
    def __init__(self):
        print('ShishiEnvExtend Environment initialized')
    def step(self):
        print('ShishiEnvExtend Step successful!')
    def reset(self):
        print('ShishiEnvExtend Environment reset')
		
if __name__ == '__main__' :
    #do some transformation.
    shishi = gym.make('CDA43_env-v0')
    