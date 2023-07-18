import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from keras.datasets import mnist

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

num_neurons = train_X.shape[1]**2
num_pattern = 3
patterns = np.zeros((num_pattern, num_neurons))
idx = np.random.permutation(np.arange(train_X.shape[0]))
patterns_m = []
for i in range(num_pattern):
    patterns[i,:] = train_X[idx[i]].flatten()
    mask = np.random.rand(train_X.shape[2], train_X.shape[2])
    mask[mask>0.3] = 0
    pattern_m = (train_X[idx[i]] * mask).flatten()
    patterns_m.append(pattern_m)

patterns = torch.tensor(patterns, dtype=torch.float)
# patterns = 2*torch.rand((num_pattern, num_neurons))-1
patterns = torch.where(patterns > 0, float(1),float(-1))



class hpf(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(patterns_m[0], dtype=torch.float), requires_grad=True)
        # self.x = nn.Parameter(2*torch.rand((num_neurons, 1), requires_grad=True)-1)
        ##### zero activation #####
        # self.x.data = torch.where(self.x.data > 0, float(1), float(-1))
        ##### tanh #####
        self.x.data = F.tanh(self.x.data)
        # self.Q = 2*torch.rand((num_neurons, num_neurons))-1
        for i, pattern in enumerate(patterns):
            if i == 0:
                self.Q = torch.outer(pattern, pattern)
            else:
                self.Q += torch.outer(pattern, pattern)
        self.Q /= num_pattern
        # self.Q.fill_diagonal_(0)
    
    def forward(self, x=None):
        return -0.5* self.x.T @ self.Q @ self.x
    


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
opt = torch.optim.Adam(quad.parameters(), lr=1e-3)

loss = []
for i in range(1000):
    y = quad()
    loss.append(y.item())
    if i % 100 == 0:
        print(i, y.item())
    idx = torch.randint(0, num_neurons, (1, ))
    # grad = torch.zeros_like(quad.x.data)
    # grad[idx] = quad.Q[idx, :] @ quad.x.data
    # y.backword(grad)
    quad.x.data[idx] = quad.Q[idx, :] @ quad.x.data
    ##### zero activation #####
    # quad.x.data[idx] = torch.where(quad.x.data[idx] > 0, float(1), float(-1))
    ##### tanh #####
    quad.x.data[idx] = F.tanh(quad.x.data[idx])
    
    # if i % 100 == 0:
    #     pattern = 2*torch.rand((num_neurons))-1
    #     pattern = torch.where(pattern > 0, float(1),float(-1))
    #     # num_pattern += 1
    #     quad.Q = (num_pattern-1)/num_pattern * quad.Q + 1/num_pattern * torch.outer(pattern, pattern)
    
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(train_X[idx[0]], cmap=plt.get_cmap('gray'))
# plt.show()
ax2.imshow(patterns_m[0].reshape(train_X.shape[2], train_X.shape[2]) , cmap=plt.get_cmap('gray'))
# plt.show()
ax3.imshow(quad.x.data.reshape(train_X.shape[2], train_X.shape[2]), cmap=plt.get_cmap('gray'))
plt.show()

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
