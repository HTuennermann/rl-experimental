import numpy as np

import datetime
class logger:
    def __init__(self, name):
        self.a = open("%s-act.txt"%name, "w")
        self.pd = open("%s-dio.txt"%name, "w")
    def diode(self, data):
        self.pd.write("%s, %E\n" % (datetime.datetime.now(), data))
    def act(self,data):
        self.a.write("%s, %E\n" % (datetime.datetime.now(), data))
    def close(self):
        self.pd.close()
        self.a.close()

#l = logger("logr")


        
# Connect the socket to the port where the server is listening
import socket
import sys
import numpy as np
import array as a
dt = np.dtype(np.float32)
# Create a TCP/IP socket

class datagrabber:    
    def __init__(self, qsize=10):
        self.qsize = qsize*2
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', 27016)
        print('connecting to %s port %s' % server_address)
        self.sock.connect(server_address)
    def send(self, d):
        
        if(d<0):
            message = a.array("f", [0]).tobytes()
        elif(d>10):
            message = a.array("f", [10]).tobytes()
        else:
            message = a.array("f", [d]).tobytes()

        self.sock.send(message)
        return (np.frombuffer(self.sock.recv(4*self.qsize),dtype=dt,count=self.qsize)).reshape([-1,2]).T


"""
Simulation module
"""
import numpy as np
import array
import gym
from gym.spaces import Box
from gym import spaces, logger

class ENV(gym.Env):
    def __init__(self, qsize=10):
        self.qsize = qsize
        #f = open(feeder, "rb")
        #input()
        self.dg = datagrabber(qsize = 10)
        
 
        self.phase = 5
        
        #self.action_space = spaces.Discrete(21)
        
        
        
    @property    
    def observation_space(self):
        return self.result #.flatten()
    
    @property
    def action_space(self):
        return Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32)
        
    def reset(self):
        self.result = self.dg.send(self.phase)         
        return self.result#.flatten()

        
    def step(self, action):
        
        a = action
        #print(a)
        rew = 0
        self.phase +=a
        if self.phase<0:
            self.phase=5

        if self.phase>10:
            self.phase=5

            
        self.result = self.dg.send(self.phase)
        
        #self.result = self.dg.send(self.phase)
        
        self.result.setflags(write=1)
        self.result[1][:-1] = np.diff(self.result[1]) ## diff only
        self.result[-1][-1] = self.phase
        #reward = self.result[0,-1]#-(self.result[-1,-1] - 0.5)**2
        reward = -(self.result[0,-1] - 0.7)**2
        #print(self.result[-1,-1])
        return self.result,reward,False,{}
    def stategrab(self):
        return self.result
    
    def seed(self, seed):
        np.random.seed(seed)

    def render(self, mode):
        #print(self.lastresult)
        pass

#e = ENV()
#input()