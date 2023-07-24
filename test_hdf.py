import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from keras.datasets import mnist

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

num_neurons = train_X.shape[1]**2
num_pattern = 2
patterns = np.zeros((num_pattern, num_neurons))
idx = np.random.permutation(np.arange(train_X.shape[0]))
patterns_m = []
for i in range(num_pattern):
    pattern = train_X[idx[i]]
    pattern = 0.5*pattern/np.max(pattern)
    patterns[i,:] = pattern.flatten()
    mask = np.random.rand(train_X.shape[2], train_X.shape[2])
    mask[mask>0.3] = 0
    pattern_m = (pattern * mask).flatten()
    patterns_m.append(pattern_m)

patterns = torch.tensor(patterns, dtype=torch.float)
patterns_m = torch.tensor(patterns_m, dtype=torch.float)
# patterns = 2*torch.rand((num_pattern, num_neurons))-1
patterns = torch.where(patterns > 0, float(1),float(-1))
patterns_m = torch.where(patterns_m > 0, float(1),float(-1))

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(patterns[0].reshape(train_X.shape[2], train_X.shape[2]), cmap=plt.get_cmap('gray'))
ax2.imshow(patterns[1].reshape(train_X.shape[2], train_X.shape[2]) , cmap=plt.get_cmap('gray'))
# ax3.imshow(quad.x.data.reshape(train_X.shape[2], train_X.shape[2]), cmap=plt.get_cmap('gray'))
plt.show()

class hpf(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(patterns_m[1], requires_grad=True)
        # self.x = nn.Parameter(2*torch.rand((num_neurons, 1), requires_grad=True)-1)
        ##### zero activation #####
        self.x.data = torch.where(self.x.data > 0, float(1), float(-1))
        ##### tanh #####
        # self.x.data = F.tanh(self.x.data)
        # self.Q = 2*torch.rand((num_neurons, num_neurons))-1
        for i, pattern in enumerate(patterns):
            if i == 0:
                self.Q = torch.outer(pattern, pattern)
            else:
                self.Q += torch.outer(pattern, pattern)
        self.Q /= num_pattern
        self.Q.fill_diagonal_(0)
    
    def forward(self, x=None):
        return -0.5* self.x.T @ self.Q @ self.x

quad = hpf()
opt = torch.optim.Adam(quad.parameters(), lr=1e-3)
num_batch = 10
loss = []
for i in range(10000):
    y = quad()
    loss.append(y.item())
    if i % 100 == 0:
        print(i, y.item())
    idx = torch.randint(0, num_neurons, (num_batch, ))
    quad.x.data[idx] = quad.Q[idx, :] @ quad.x.data
    ##### zero activation #####
    quad.x.data[idx] = torch.where(quad.x.data[idx] > 0, float(1), float(-1))
    ##### tanh #####
    # quad.x.data[idx] = F.tanh(quad.x.data[idx])
    
    
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(patterns[1].reshape(train_X.shape[2], train_X.shape[2]), cmap=plt.get_cmap('gray'))
ax2.imshow(patterns_m[1].reshape(train_X.shape[2], train_X.shape[2]) , cmap=plt.get_cmap('gray'))
ax3.imshow(quad.x.data.reshape(train_X.shape[2], train_X.shape[2]), cmap=plt.get_cmap('gray'))
plt.show()

fig = plt.figure()
loss_ax = fig.add_subplot(111)
loss_ax.plot(loss)
plt.show()

