"""
Simulation module
"""
import numpy as np
import array
from gym.spaces import Box

class ENV():
    def __init__(self):
        f = open("/Users/henrik/decimate2.bin", "rb")
        self.phase = 0.
        self.last = 0.
        
        self.data = np.array(array.array("d", f.read(209715200))) 
        self.index = np.random.randint(100000)
        self.act = np.zeros(10)
        self.pd = np.zeros(10)
        
        
    @property    
    def observation_space(self):
        return np.vstack([self.act,self.pd]).flatten()
    
    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        
    def reset(self):
        self.index = np.random.randint(100000)+10
        self.act = np.zeros(10)
        self.pd = np.cos(self.data[self.index-10:self.index])**2
        
        self.phase = 0
        return np.vstack([self.act,self.pd]).flatten()

        
    def step(self, action):
        #print("action", action)
        if np.isnan(action[0]):
            action=0
            print(self.last)
            raise(ValueError)
        else:
            action = action[0]

        self.phase +=action
        result = np.cos(self.data[self.index]+self.phase)**2
        #print "result" , result
        self.act = np.roll(self.act,1)
        self.pd = np.roll(self.pd,1)
        self.act[0] = action
        self.pd[0] = result
        self.index += 1
        reward = result
        #reward = -np.sqrt((result-0.5)**2)
        self.lastresult = reward
        #print "reward" , reward


        #reward-=(action)**2/10

       
 

        if self.index > 200000:
            return np.vstack([self.act,self.pd]).flatten(),reward,True,{}

        self.last = np.vstack([self.act,self.pd]).flatten(),reward,False,{}
        return np.vstack([self.act,self.pd]).flatten(),reward,False,{}
    
    def seed(self, seed):
        np.random.seed(seed)

    def render(self, mode):
        #print(self.lastresult)
        pass
        
        

if __name__ == "__main__":
    env = ENV()
    env.reset()
    print(env.action_space)