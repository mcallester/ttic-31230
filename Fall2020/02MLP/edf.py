import numpy as np

class Model:
    def __init__(self, loss):
        self.components = []
        self.loss =loss
        loss.walk(self.components)
        for i,c in list(enumerate(self.components)):
            c.index = i

    def forward(self):
        for c in self.components: c.forward()

    def backward(self):
        for c in self.components: c.grad.fill(0)
        self.loss.grad.fill(1.0)
        for c in reversed(self.components): c.backward()

    def update(self):
        for c in self.components:
            if isinstance(c,Parameter): c.update()

    def backprop(self):
        self.forward()
        self.backward()
        self.update()
                
class Component:
    def __init__(self):
        self.inputs = []
        self.users = []
        if hasattr(self, 'value'): self.shape = self.value.shape
        self.grad = np.empty(self.shape,np.float32)
        self.walked = False

    def walk(self,components):
        if not self.walked:
            for c in self.inputs:
                c.walk(components)
            components.append(self)
            self.walked = True

    def __str__(self):
        return format('Component {} {}'.format(self.index, type(self)))

    def __repr__(self):
        return self.__str__()

def input_of(x,y):
    x.users.append(y)
    y.inputs.append(x)

class Input(Component):

    """An input is initialized with a shape but not a value.
    The value of an input is set in the training loop"""

    def __init__(self,shape):
        if not isinstance(shape,tuple):
            raise ValueError('illegal input shape --- the shape must be a tuple')
        self.shape = shape
        Component.__init__(self)
    def forward(self):None
    def backward(self):None

class Parameter(Input):

    """Parameters are also Inputs but the value is set at circuit creation time
    and updated duting training"""
    
    def __init__(self,x):
        self.value = x
        Component.__init__(self)

    def update(self):
        if hasattr(self, '__update__'):
            self.__update__(self)
        else: default_update(self)

"""model verification"""

def verify(M, epsilon = .001):

    """this compares backpropagation to numerical differentiation
    at every component of a Model.  This is used to debug the backward method
    of a component implementation and should used on small circuits."""
    
    M.forward()
    M.backward()

    V0 = np.sum(M.loss.value)
    for i, c in reversed(list(enumerate(M.components))):
        print(' ')
        print('component ', i, type(c))
        print(format('users: {}'.format([u.index for u in c.users])))
        print('value: ')
        print(c.value)
        print('backprop gradient: ')
        print(c.grad)
        
        size = np.prod(c.shape)
        numgrad = np.empty(size)
        linval = c.value.reshape(size)
        
        for i in range(np.prod(c.shape)):
            oldval = linval[i]
            linval[i] = oldval + epsilon
            forward_users(c) #this is potentially exponential in large circuits
            numgrad[i] = (np.sum(M.loss.value) - V0)/epsilon
            linval[i] = oldval
            forward_users(c) #this sets the user values back
            
        print('numerical gradient:')
        print(numgrad.reshape(c.shape))

def forward_users(c):
    for user in c.users:
        user.forward()
        forward_users(user)
        
""" ***************** Some Components ************* """

class Norm(Component):

    """for y = Norm(x) we create new parameters alpha and beta with the same shape as x and then
    y[i1,...,ik] = alpha[i1,...ik]x[i1,...,ik] + beta[i1,...,ik]"""
    
    def __init__(self, x):
        self.value = np.empty(x.shape)
        alpha = Parameter(np.ones(x.shape)*.1)
        beta = Parameter(np.zeros(x.shape))
        self.x = x
        self.alpha = alpha
        self.beta = beta
        Component.__init__(self)
        input_of(x,self)
        input_of(alpha,self)
        input_of(beta,self)
        
    def forward(self):
        np.add(self.x.value * self.alpha.value, self.beta.value, self.value)

    def backward(self):
        np.add(self.grad * self.alpha.value, self.x.grad, self.x.grad)
        np.add(self.grad, self.beta.grad, self.beta.grad)
        np.add(self.grad * self.x.value, self.alpha.grad, self.alpha.grad)


class Sigmoid(Component):

    """for y = Sigmoid(x) we have that y has the same shape
    as x where y[i1,...,ik] = sigmoid(x[i1,...ik]).
    We typically have that x and y have shape (B,I)
    where B is the minibatch size."""

    def __init__(self,x):
        self.value = np.empty(x.shape,np.float32)
        self.x = x
        Component.__init__(self)
        input_of(x,self)
        self.negexp = np.empty(self.shape)
        
    def forward(self):
        np.exp(np.negative(self.x.value), self.negexp)
        np.reciprocal(np.add(self.negexp, 1), self.value)

    def backward(self):
        np.add(self.grad * np.square(self.value) * self.negexp,
               self.x.grad,
               self.x.grad)

class NegLog(Component):

    """for ell = NegLog(p) we have that y has the same shape
    as x where ell[i1,...,ik] = - log(x[i1,...ik]).
    We typically have that ell and p have shape (B,I)
    where B is the minibatch size.  This is typically used as a
    loss function"""

    def __init__(self,p):
        self.value = np.empty(p.shape)
        self.p = p
        Component.__init__(self)
        input_of(p,self)
        
    def forward(self):
        np.negative(np.log(self.p.value), self.value)

    def backward(self):
        np.add(- self.grad / self.p.value,
               self.p.grad,
               self.p.grad)

class VDot(Component):

    """Matrix vector product: For y = VDot(A,x) we assume that A has shape (I,J) and
    x has shape J in which case y has shape (I) with y[i] = sum_j  A[i,j] x[j]."""
    
    def __init__(self,A,x):
        if not isinstance(A,Parameter):
            raise ValueError('A is not a parameter in VDot')
        if isinstance(x,Parameter):
            raise ValueError('x is a parameter in VDot')
        if not (len(A.shape) == 2 and len(x.shape) == 1 and A.shape[1] == x.shape[0]):
            raise ValueError('incompatible shapes in VDot')
        self.value = np.empty((A.shape[0],))
        self.A = A
        self.x = x
        Component.__init__(self)
        input_of(A,self)
        input_of(x,self)

    def forward(self):
        np.dot(self.A.value, self.x.value, out=self.value)

    def backward(self):
        np.add(np.outer(self.grad, self.x.value),
               self.A.grad,
               self.A.grad)
        np.add(np.dot(self.grad,self.A.value),
               self.x.grad,
               self.x.grad)

class Aref(Component):

    """ for y = Aref(x,i) we require that x has shape (N) and i has shape (1)
    in which case y has shape (1) with y[0] = x[,i[0]].
    In general scalars will be treated as tensors of shape (1)."""

    def __init__(self,a,i):
        if not (len(a.shape) == 1 and len(i.shape) == 1):
            raise ValueError("incompatible shapes in Aref")

        self.value = np.empty(1)
        self.a = a
        self.i = i
        Component.__init__(self)
        input_of(a,self)
        input_of(i,self)

    def forward(self):
        self.value[0] = self.a.value[self.i.value[0]]

    def backward(self):
        self.a.grad[self.i.value[0]] = self.a.grad[self.i.value[0]] + self.grad[0]


class Softmax(Component):

    """for y = Softmax(x) we have that y is the softmax of x over the last index of x.
    y[b,i] = (1/Z[b]) exp(x[b,i])  where Z[b] =- sum_i exp(x[b,i])
    we can construct a softmax over another index using a transposition view."""

    def __init__(self,x):
        self.value = np.empty(x.shape)
        self.x = x
        Component.__init__(self)
        input_of(x,self)
        
    def forward(self):
        exp = np.exp(self.x.value)
        z = np.sum(exp,len(self.x.shape)-1)
        new_shape = list(self.x.shape)
        new_shape[len(new_shape)-1] = 1
        np.divide(exp, z.reshape(new_shape), self.value)

    def backward(self):
        np.subtract(self.x.grad, np.dot(self.grad,self.value)*self.value, self.x.grad)
        np.add(np.multiply(self.grad,self.value), self.x.grad, self.x.grad)
        
""" ****************** update methods ********************"""

lmb = .01

def default_update(self):
    np.subtract(self.value, lmb * self.grad, self.value)
