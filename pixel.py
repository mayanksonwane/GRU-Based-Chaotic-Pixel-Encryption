import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools
import pickle

%autosave 180


####  Define the network parameters:
hiddenSize = 1 # network size, this can be any number (depending on your task)
numClass = 1 # this is the same as our vocab_size
vocab_size=1
batch_size=1
#### Weight matrices for our inputs
Wz = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)
Wr = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)
Wh = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)

## Intialize the hidden state
# this is for demonstration purposes only, in the actual model it will be initiated during training a loop over the
# the number of bacthes and updated before passing to the next GRU cell.
h_t_demo = torch.zeros(batch_size, hiddenSize)

#### Weight matrices for our hidden layer
Uz = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)
Ur = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)
Uh = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)

#### bias vectors for our hidden layer
bz = Variable(torch.zeros(hiddenSize), requires_grad=True)
br = Variable(torch.zeros(hiddenSize), requires_grad=True)
bh = Variable(torch.zeros(hiddenSize), requires_grad=True)

#### Output weights
Wy = Variable(torch.randn(hiddenSize, numClass), requires_grad=True)
by = Variable(torch.zeros(numClass), requires_grad=True)

print(f'Weight Matrix for Wz:\n {Wz,Wz.size()}','\n')
print(f'Weight Matrix for Uz:\n {Uz, Uz.size()}','\n')
print(f'Weight Matrix for h_t_demo:\n {h_t_demo,h_t_demo.size()}','\n')
print(f'Bias vector for bz: \n {bz, bz.size()}','\n')
