import numpy as np
cimport numpy as np
import random
import sys
import math
import theano
import theano.tensor as T
import time

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def softmax(param):
    return np.exp(param)/np.exp(param).sum()

def find_reasonable_epsilon(parameter, energy_function, energy_function_delta, argument):
    cdef float epsilon = 5e-3
    cdef np.ndarray momentum = np.random.normal(0., 1., len(parameter))
    cdef float hamilton = energy_function(parameter, argument) - np.sum(momentum**2.0) / 2.0
    current_parameter_condinate, momentum = leapfrog(parameter, momentum, epsilon, energy_function_delta, argument)
    cdef float differenceHamilton = (energy_function(current_parameter_condinate, argument) - np.sum(momentum ** 2.0) / 2.0)  - hamilton
    #print differenceHamilton
    #sys.exit(1)
    cdef float accept_probability = np.exp(differenceHamilton)
    

    a = 2 * int( accept_probability > 0.5) - 1

    while accept_probability ** a > 2 **-a :
        
        epsilon = 2. ** a * epsilon
        current_parameter_condinate, momentum = leapfrog(current_parameter_condinate, momentum, epsilon, energy_function_delta, argument)
        differenceHamilton = (energy_function(current_parameter_condinate, argument) - np.sum(momentum ** 2.0) / 2.0) - hamilton
        accept_probability = np.exp(differenceHamilton)
        
    print "find_reasonable_epsilon=", epsilon ** a
    #sys.exit(1)
    return epsilon

cdef leapfrog(np.ndarray current_parameter_condinate,np.ndarray momentum,float step_accuracy,energy_function_delta,argument):
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate, argument)
    current_parameter_condinate = current_parameter_condinate + step_accuracy * momentum
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate, argument)
    return current_parameter_condinate, momentum
    
def hamiltonianMonteCarlo(np.ndarray parameter, energy_function, energy_function_delta, argument=None, iter=10000,bear_in=9000, iter_leapfrog=20):
    cdef np.ndarray current_parameter_condinate
    cdef np.ndarray momentum
    cdef float hamilton
    cdef float differenceHamilton
    cdef float accept_probability 
    cdef float H_t
    cdef float w
    cdef float average_H
    cdef float step_size
    cdef int _

    cdef float target_accept = 0.8
    cdef float t0 = 10
    cdef float gamma = 0.05
    average_H = 0.0
    
    cdef np.ndarray current_parameter = parameter
    cdef list r = []
    
    step_size  = find_reasonable_epsilon(parameter, energy_function,energy_function_delta, argument)
    u = np.log(10 * step_size)
    
    
    cdef int i
    for i in range(iter):
        
        

        momentum = np.random.randn(len(parameter))
        hamilton = energy_function(current_parameter, argument) - np.sum(momentum**2.0) / 2.0
        current_parameter_condinate = current_parameter
        
        for _ in range(iter_leapfrog):
            current_parameter_condinate, momentum = leapfrog(current_parameter_condinate,momentum,step_size,energy_function_delta, argument)
            
        
        differenceHamilton = ( (energy_function(current_parameter_condinate, argument) - np.sum(momentum ** 2.0) / 2.0) ) - hamilton

        accept_probability = min(1.,np.exp(differenceHamilton))
        if i < bear_in:
            H_t = target_accept - accept_probability
            w = 1. / ( i + t0 )
            average_H = (1 - w) * average_H + w * H_t
            step_size = min(10.,np.exp(u - (np.sqrt(i)/gamma)*average_H))
        
            if i % 100 == 0:
                print "------ %d ------"%(i)
                print "accept_probability:{}".format(accept_probability)
                print "average_H:{}".format(average_H)
                print "step_size:{}".format(step_size)

        if random.random() < np.exp(differenceHamilton):
            current_parameter = current_parameter_condinate

        
        if i % 1000 == 0:
            print "====== %d ======"%(i)
            print "{0}".format(softmax(current_parameter))
            print "likelyfood:%f"%(energy_function(current_parameter_condinate, argument))

        if i > bear_in and i % 10 == 0:
            r.append(current_parameter)

    return np.average(r,axis=0)
def bern(p):
    return np.random.uniform() < p    
def callPosterior(parameter,argument):
    return argument[0](parameter,0,5)
def callGPosterior(parameter,argument):
    return argument[1](parameter,0,5)
def test_hmc():
    np.random.seed(1234567)
    x = T.dvector('x')
    u = T.dscalar('u')
    sigma = T.dscalar('sigma')
    n = theano.shared(np.array([1000,10000,5000,2000]), name='n')
    normalPdfSyntax = - (x.shape[0] / 2.) * T.log( 2. * math.pi * sigma ** 2) - (1./(2*sigma ** 2)) * T.sum((x-u) ** 2)
    posteriorSyntax = T.sum(n * T.log(T.nnet.softmax(x))) + normalPdfSyntax
    posterior = theano.function(inputs=[x,u,sigma], outputs=posteriorSyntax)

    gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=x)
    gPosterior = theano.function(inputs=[x,u,sigma], outputs=gPosteriorSyntax)

    estimated_parameter = hamiltonianMonteCarlo(np.random.normal(0., 1., 4),callPosterior,callGPosterior,argument=[posterior,gPosterior])

    print softmax(estimated_parameter)


