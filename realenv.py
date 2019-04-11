from __future__ import division
import numpy as np

import nidaqmx

import numpy as np
import array
import gym
from gym.spaces import Box
from gym import spaces, logger

import datetime
class logger:
    def __init__(self, name):
        self.a = open("%s-act.txt"%name, "w")
        self.pd = open("%s-dio.txt"%name, "w")
    def diode(self, data):
        self.pd.write("%s, %E, %E\n" % (datetime.datetime.now(), data[0], data[1]))
    def act(self,data):
        self.a.write("%s, %E\n" % (datetime.datetime.now(), data))
    def close(self):
        self.pd.close()
        self.a.close()

#l = 
#l.diode(1)
        
class daq:
    def __init__(self, logname="testrun"):
        self.out = nidaqmx.Task()
        self.inp = nidaqmx.Task()
        self.out.ao_channels.add_ao_voltage_chan("Dev2/ao0", min_val=0, max_val=5.0)
        self.inp.ai_channels.add_ai_voltage_chan("Dev2/ai0:1")
        self.l = logger(logname)

    def move(self,act):
        if(act>0 and act<5):
            self.out.write(act)
            self.l.act(act)
            
    def sense(self):
        d = np.average(np.array(self.inp.read(number_of_samples_per_channel=10)), axis=1)

        self.l.diode(d)
        return d[0]


class ENV(gym.Env):
    def __init__(self, logname="testrun"):
        #f = open(feeder, "rb")

        self.act = np.zeros(10)
        self.pd = np.zeros(10)
        
        self.d = daq(logname)
        self.phase = 2.5
        
        #self.action_space = spaces.Discrete(21)
        
        
    @property    
    def observation_space(self):
        return np.vstack([self.act,self.pd,]).flatten()
    
    @property
    def action_space(self):
        return Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32)
        
    def reset(self):

        self.act = np.zeros(10)
        self.pd = np.zeros(10)
        

        
        #
        self.d.move(self.phase)
        
        for i in range(10):
            self.pd[9-i] = self.d.sense()
            
        
        return np.vstack([self.act,self.pd,]).flatten()

        
    def step(self, action):
        
        if np.isnan(action[0]):
            action=0
            print(self.last)
            raise(ValueError)
        else:
            action = action[0]

        #print(a)
        rew = 0
        self.phase +=action
        if self.phase<0:
            self.phase=2.5
            rew = -50
        if self.phase>5:
            self.phase=2.5
            rew = -50
            
        self.d.move(self.phase)
        result = self.d.sense() 
        
        
        
        #print "result" , result
        self.act = np.roll(self.act,1)
        
        self.pd = np.roll(self.pd,1)
        self.act[0] = action*100
        #self.act[-1] = self.phase
        self.pd[0] = result
        #self.index += 1
        #reward = (result)**4-a#-(result-0.5)**2
        reward = -(result-0.8333431500000001)**2 *100
        #reward = result #- self.pd.std()*10.0
        #reward = np.min(self.pd)
        self.lastresult = reward
        #reward = reward + rew

 



        self.last = np.vstack([self.act,self.pd]).flatten(),reward,False,{}
        return np.vstack([self.act,self.pd]).flatten(),reward,False,{}
    def stategrab(self):
        return np.vstack([self.act,self.pd]).flatten()
    
    def seed(self, seed):
        np.random.seed(seed)

    def render(self, mode):
        #print(self.lastresult)
        pass