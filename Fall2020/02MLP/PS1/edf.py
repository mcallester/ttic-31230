import numpy as np

DT=np.float32
learning_rate = .001

############################################ Clearing the Computation graph #################################        
# This should be done immediately before creating a new computation graph.

def clear_compgraph():
    global CompNodes, Parameters
    CompNodes = []
    Parameters = []

############################################ Forward-Backward and SGD #################################

def Forward():
    for c in CompNodes: c.forward()

def Backward(loss):
    for c in CompNodes + Parameters:
        c.grad = np.zeros(c.value.shape, dtype = DT)
    loss.grad = np.ones(loss.value.shape)/len(loss.value)  #The convention is to compute the averaage gradient over the batch
    for c in CompNodes[::-1]:
        c.backward();

def SGD():
    for p in Parameters: p.SGD()

############################################ Computation Graphs #################################

"""
There are three kinds of nodes in a computation graph, inputs, parameters, and computed nodes (CompNodes).
Computed nodes are defined on a case by case bases.  The Sigmoid class is defined as an example.  Other classes are
defined below.
"""

class Input:
    def __init__(self):
        pass

    def addgrad(self, delta):
        pass
        
class Parameter:
    def __init__(self,value):
        self.value = DT(value)
        Parameters.append(self)

    def addgrad(self,delta):
        self.grad += np.sum(delta, axis = 0)

    def SGD(self):  #this is a default
        self.value -= learning_rate*self.grad

class CompNode:
    def addgrad(self, delta):
        self.grad += delta

############### Parameter Packages and some Compnodes ###########################

class ParameterPackage:
    pass
    
class AffineParams(ParameterPackage):
    def __init__(self,nInputs,nOutputs):
        X = Xavier(nInputs)
        self.w = Parameter(np.random.uniform(-X,X,(nInputs,nOutputs)))
        self.b = Parameter(np.zeros(nOutputs))

def Xavier(nInputs):
    return np.sqrt(3.0/nInputs)

class Affine(CompNode):
    def __init__(self,Phi,x):
        CompNodes.append(self)
        self.x = x
        self.Phi = Phi

    def forward(self):
        self.value = np.matmul(self.x.value,self.Phi.w.value) + self.Phi.b.value # the addition broadcasts b over the batch.

    def backward(self):
        self.x.addgrad(np.matmul(self.grad,self.Phi.w.value.transpose()))
        self.Phi.b.addgrad(self.grad)
        self.Phi.w.addgrad(self.x.value[:,:,np.newaxis] * self.grad[:,np.newaxis,:])

class Sigmoid(CompNode):
    def __init__(self,x):
        CompNodes.append(self)
        self.x = x

    def forward(self):
        bounded = np.maximum(-10,np.minimum(10,self.x.value)) #blocks numerical warnings
        self.value = 1 / (1 + np.exp(-bounded))

    def backward(self):
        self.x.addgrad(self.grad * self.value * (1-self.value))

class Softmax(CompNode):
    def __init__(self,s): #s has shape (nBatch,nLabels}
        CompNodes.append(self)
        self.s = s
        
    def forward(self):
        smax = np.max(self.s.value,axis=1,keepdims=True)
        bounded = np.maximum(-10,self.s.value - smax) #blocks numerical warnings
        es = np.exp(bounded) 
        self.value = es / np.sum(es,axis=1,keepdims=True)

    def backward(self):
        p_dot_pgrad = np.matmul(self.value[:,np.newaxis,:],self.grad[:,:,np.newaxis]).squeeze(-1) # p dot p.grad with shape (nBatch,1)
        self.s.addgrad(self.value * (self.grad - p_dot_pgrad)) # p_dot_pgrad is broadcast over the labels

class LogLoss(CompNode):
    def __init__(self,p,y):
        if not isinstance(p,CompNode):
            print('LogLoss probability vector is not a CompNode')
            raise ValueError
        CompNodes.append(self)
        self.p = p
        self.y = y

    def forward(self):
        dLabels = self.p.value.shape[1]
        dBatch = len(self.y.value)
        self.picker = np.arange(dBatch)*dLabels + self.y.value  # p.flatten[picker][b] =p[b,y[b]] where b is the batch index
        self.value = - np.log(self.p.value.reshape((-1,))[self.picker])
        
    def backward(self):
        flatpval = self.p.value.reshape((-1,))
        flatpgrad = self.p.grad.reshape((-1,))
        flatpgrad[self.picker] -= self.grad/flatpval[self.picker]

