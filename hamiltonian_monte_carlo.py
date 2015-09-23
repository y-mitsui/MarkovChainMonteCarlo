import numpy
import random
import sys
import theano
import theano.tensor as T
import time

def softmax(param):
    return numpy.exp(param)/numpy.exp(param).sum()
    
def find_reasonable_epsilon(parameter, energy_function, energy_function_delta):
    epsilon = 1e-1
    momentum = numpy.random.normal(0., 1., len(parameter))
    hamilton = energy_function(parameter) - numpy.sum(momentum**2.0) / 2.0
    current_parameter_condinate, momentum = leapfrog(parameter, momentum, epsilon, energy_function_delta)
    differenceHamilton = (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0)  - hamilton
    accept_probability = numpy.exp(differenceHamilton)
    a = 2 * int( accept_probability > 0.5) - 1

    while accept_probability ** a > 2 **-a :
        
        epsilon = 2. ** a * epsilon
        current_parameter_condinate, momentum = leapfrog(current_parameter_condinate, momentum, epsilon, energy_function_delta)
        differenceHamilton = (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0) - hamilton
        accept_probability = numpy.exp(differenceHamilton)
        
    print "find_reasonable_epsilon=", epsilon ** a

    return epsilon

def leapfrog(current_parameter_condinate,momentum,step_accuracy,energy_function_delta):
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate)
    current_parameter_condinate = current_parameter_condinate + step_accuracy * momentum
    momentum = momentum + (step_accuracy/2.) * energy_function_delta(current_parameter_condinate)
    return current_parameter_condinate, momentum     
    
def hamiltonianMonteCarlo(parameter, energy_function, energy_function_delta, iter=10000,bear_in=5000, iter_leapfrog=30, leapfrog_accuracy=0.001):

    target_accept = 0.8
    t0 = 10
    n = 2000
    step_scale=0.25
    step_size = step_scale / n**(1/4.)
    gamma = 0.05
    flg = True
    u = numpy.log(10 * step_size)
    average_H = 0.0
    
    current_parameter = parameter
    r = []
    step_accuracy=leapfrog_accuracy / 2.
    avg_prob1 = 0.
    diff_avg0 = 0.
    Dec=True
    steps = [0.001 * 2**i for i in range(100)]
    
    step_size  = find_reasonable_epsilon(parameter, energy_function,energy_function_delta)
    u = numpy.log(10 * step_size)
    
    for i in range(iter):
        momentum = numpy.random.randn(len(parameter))
        hamilton = energy_function(current_parameter) - numpy.sum(momentum**2.0) / 2.0
        current_parameter_condinate = current_parameter
        
        for _ in range(iter_leapfrog):
            current_parameter_condinate, momentum = leapfrog(current_parameter_condinate,momentum,step_size,energy_function_delta)
            
        
        differenceHamilton = ( (energy_function(current_parameter_condinate) - numpy.sum(momentum ** 2.0) / 2.0) ) - hamilton

        accept_probability = min(1.,numpy.exp(differenceHamilton))
        #print "accept_probability:{}".format(accept_probability)
        H_t = target_accept - accept_probability
        w = 1. / ( i + t0 )
        average_H = (1 - w) * average_H + w * H_t
        #print u-(numpy.sqrt(i)/gamma)*average_H
        step_size = numpy.exp(u - (numpy.sqrt(i)/gamma)*average_H)
        
        if i % 100 == 0:
            print "------ %d ------"%(i)
            print "accept_probability:{}".format(accept_probability)
            print "average_H:{}".format(average_H)
            print "step_size:{}".format(step_size)
            
        if random.random() < numpy.exp(differenceHamilton):
            current_parameter = current_parameter_condinate

        
        if i % 1000 == 0:
            print "%d"%(i)
            print "{0}".format(softmax(current_parameter))
            print "likelyfood:%f"%(energy_function(current_parameter_condinate))
        if i > bear_in and i % 10 == 0:
            r.append(softmax(current_parameter))
        #time.sleep(0.5)
    print numpy.average(r,axis=0)
    return current_parameter
def bern(p):
    return numpy.random.uniform() < p    
def callPosterior(parameter):
    #print posterior([2,2,2],parameter,0,5)
    return posterior([20,20,20],parameter)
def callGPosterior(parameter):
    #print gPosterior([2,2,2],parameter,0,5)
    return gPosterior([20,20,20],parameter)
    
x = T.dvector('x')
nSample = T.dvector('nSample')
#normalPdfSyntax = 1. / T.sqrt(2 * 3.14159 * sigma ** 2) * T.exp(-(x-u) ** 2 / (2 * sigma ** 2))
#posteriorSyntax = -( (T.sum(nSample * T.log(T.nnet.softmax(x))) + T.sum(T.log(normalPdfSyntax))) )
posteriorSyntax = T.sum(nSample * T.log(T.nnet.softmax(x)))
posterior = theano.function(inputs=[nSample,x], outputs=posteriorSyntax)
#print posterior([2,2,1000],[1,2,3],0,5)

gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=x)
gPosterior = theano.function(inputs=[nSample,x], outputs=gPosteriorSyntax)
#print gPosterior([2,2,1000],[1,2,3],0,5)


estimated_parameter = hamiltonianMonteCarlo(numpy.random.normal(0., 1., 3),callPosterior,callGPosterior)

print softmax(estimated_parameter)


