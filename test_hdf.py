import torch
import torch.nn as nn
import matplotlib.pyplot as plt

num_neurons = 3
num_pattern = 10
patterns = torch.rand((num_pattern, num_neurons))

class hpf(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(2*torch.rand((num_neurons, 1), requires_grad=False)-1)
        self.Q = 2*torch.rand((num_neurons, num_neurons))-1
        # for i, pattern in enumerate(patterns):
        #     if i == 0:
        #         self.Q = torch.outer(pattern, pattern)
        #     else:
        #         self.Q += torch.outer(pattern, pattern)
        # self.Q /= num_pattern
    
    def forward(self, x=None):
        return -self.x.T @ self.Q @ self.x
    


class hpf_opt(torch.optim.Optimizer):
    # Init Method:
    def __init__(self, params, lr=1e-3):
        super(hpf_opt, self).__init__(params, defaults={'lr': lr})
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
      
    # Step Method
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom
    

# x = torch.rand((3, 1), requires_grad=True)
# Q = torch.rand((3, 3))
# y = x.T @ Q @ x

quad = hpf()
# opt = torch.optim.Adam(quad.parameters(), lr=1e-3)

loss = []
for i in range(10000):
    quad.x.data = quad.Q @ quad.x.data
    quad.x.data = torch.where(quad.x.data > 0, float(1), float(-1))
    y = quad()
    loss.append(y.item())
    if i % 1000 == 0:
        print(i, y.item())


fig = plt.figure()
loss_ax = fig.add_subplot(111)
loss_ax.plot(loss)
plt.show()

#imports
import numpy as np
#for visualization
import matplotlib.pyplot as plt

class Hopfield_Net: #network class
    #init ialize network variables and memory
    def __init__(self,input):
        
        #patterns for network training / retrieval
        self.memory = np.array(input)
        #single vs. multiple memories
        if   self.memory.size > 1:
             self.n = self.memory.shape[1] 
        else:
             self.n = len(self.memory)
        #network construction
        self.state = np.random.randint(-2,2,(self.n,1)) #state vector
        self.weights = np.zeros((self.n,self.n)) #weights vector
        self.energies = [] #container for tracking of energy
        
       
    def network_learning(self): #learn the pattern / patterns
        self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory #hebbian learning
        np.fill_diagonal(self.weights, 0)


    def update_network_state(self,n_update): #update network
        for neuron in range(n_update): #update n neurons randomly
            self.rand_index = np.random.randint(0,self.n) #pick a random neuron in the state vector
            #Compute activation for randomly indexed neuron
            self.index_activation = np.dot(self.weights[self.rand_index,:],
                                           self.state) 
            #threshold function for binary state change
            if self.index_activation < 0: 
                self.state[self.rand_index] = -1
            else:
                self.state[self.rand_index] =  1

            
    def compute_energy(self): #compute energy
        self.energy = -0.5*np.dot(np.dot(self.state.T,self.weights),self.state)
        self.energies.append(self.energy)
