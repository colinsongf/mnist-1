import mnist, numpy, time
from scipy.stats import bernoulli
from scipy.special import expit

# random_params: (int, int, tuple, tuple) -> tuple
def random_params(nvisible, nhidden, loc, scale):
    return (numpy.random.normal(loc=loc[0],scale=scale[0],size=nvisible), numpy.random.normal(loc=[1],scale=scale[1],size=nhidden), numpy.random.normal(loc=loc[2],scale=scale[2],size=(nvisible,nhidden)))

# hidden_fields: (array, array, vector) -> array
def hidden_fields(visible, W, b):
    return numpy.reshape(b, (1,len(b))) + numpy.dot(visible,W)

# visible_fields: (array, array, vector) -> array
def visible_fields(hidden, W, a):
    return numpy.reshape(a, (1,len(a))) + numpy.dot(hidden,W.T)
    
# logistic: (array) -> array
def logistic(anarray):
    return expit(anarray)

# random_logistic: (array) -> array
def random_logistic(anarray):
    return bernoulli.rvs(logistic(anarray))
    
# gibbs_forward: (array, array, vector) -> array
def gibbs_forward(v_data,W,b):
    return random_logistic(hidden_fields(v_data,W,b))
    
# gibbs_backward: (array, array, vector) -> array
def gibbs_backward(hidden,W,a):
    return random_logistic(visible_fields(hidden,W,a))
    
# gibbs: (array, vec, vec, array, int) -> tuple
def gibbs(v_data, a, b, W, steps):
    h_data = gibbs_forward(v_data, W, b)
    h_free = h_data.copy()    
    for i in range(steps):
        v_free = gibbs_backward(h_free,W,a)
        h_free = gibbs_forward(v_free,W,b)
    return (h_data, v_free, h_free)
    
# rmse: (array, array) -> float
def rmse(v_data, v_free):
    return numpy.sqrt(numpy.mean(numpy.ravel(v_data - v_free)**2))
    
# grad_a: (array, array) -> vec
def grad_a(v_data, v_free):
    return numpy.mean(v_data - v_free, axis=0)
    
# grad_a: (array, array) -> vec
def grad_b(h_data, h_free):
    return numpy.mean(h_data - h_free, axis=0)
    
# grad_W: (array, array, array, array) -> array
def grad_W(h_data, v_data, h_free, v_free):
    return (numpy.dot(v_data.T,h_data) - numpy.dot(v_free.T,h_free))/len(v_data)
    
# descent: (array, int, tuple, int, int, float, int, OPTIONAL) -> tuple
def descent(images, batchsize, dim, n_hidden, n, momentum, epochs, method = "RMSprop"):
    astep, bstep, Wstep = 0.001, 0.001, 0.001
    a,b,W = random_params(dim[0]*dim[1], n_hidden, (-0.5, -0.2, 0.0), (0.05, 0.05, 0.5))
    Da, Db, DW = numpy.zeros_like(a), numpy.zeros_like(b), numpy.zeros_like(W) 
    MSa, MSb, MSW = numpy.zeros_like(a), numpy.zeros_like(b), numpy.zeros_like(W) 
    # an array to store the reconstruction error values during the descent
    mem = []
    
    # permute the images into a random order
    randim = mnist.random_permute(images)
    for epoch in range(epochs):
        #learning rate decays slowly duing descent
        lr = 0.8**epoch
            
        for t in range( int(len(randim)/batchsize) ):
            # grab a minibatch and take a n Monte Carlo steps
            v_data = mnist.stochastic_binarize(randim[batchsize*t:batchsize*t + batchsize], 255)    
            h_data, v_free, h_free = gibbs(v_data, a, b, W, n)
            if t % 10 == 0:
                err = rmse(v_data, v_free)
                print("{0}: {1}: {2:.4f}".format(epoch,t,err))
                mem.append(err)        
            
            # compute the gradients
            da, db, dW = grad_a(v_data, v_free), grad_b(h_data, h_free), grad_W(h_data, v_data, h_free, v_free)

            # compute the updates using RMSprop (slide 29 lecture 6 of Geoff Hinton's coursera course) or momentum
            if method == "RMSprop":
                MSa = 0.9*MSa + (1-0.9)*da**2
                MSb = 0.9*MSb + (1-0.9)*db**2
                MSW = 0.9*MSW + (1-0.9)*dW**2
                
                Da = lr*astep*da / (0.00001 + numpy.sqrt(MSa))   
                Db = lr*bstep*db / (0.00001 + numpy.sqrt(MSb))  
                DW = lr*Wstep*dW / (0.00001 + numpy.sqrt(MSW))
            elif method == "momentum":
                Da = lr*astep*da + momentum*Da   
                Db = lr*bstep*db + momentum*Db     
                DW = lr*Wstep*dW + momentum*DW    
            else:
                raise ValueError("method must be one of: RMSprop, momentum")
                
            # update the paramters
            a, b, W = a + Da, b + Db, W + DW
    
    return (a, b, W, mem)

if __name__ == "__main__":
    pass

