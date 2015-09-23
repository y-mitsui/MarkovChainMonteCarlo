import numpy
import random
import sys
import theano
import theano.tensor as T

def softmax(param):
    return numpy.exp(param)/numpy.exp(param).sum()
    
def hamiltonianMonteCarlo(parameter, energy_function, energy_function_delta, iter=2000,bear_in=1000, iter_leapfrog=30, leapfrog_accuracy=0.001):
    current_parameter = parameter
    r = []
    step_accuracy=leapfrog_accuracy / 2.
    for i in range(iter):
        momentum = numpy.random.randn(len(parameter))
        hamilton = numpy.sum(momentum**2.0) / 2.0 - energy_function(current_parameter)
        current_parameter_condinate = current_parameter
        
        for _ in range(iter_leapfrog):
            momentum = momentum + step_accuracy * energy_function_delta(current_parameter_condinate)
            current_parameter_condinate = current_parameter_condinate + leapfrog_accuracy * momentum
            momentum = momentum + step_accuracy * energy_function_delta(current_parameter_condinate)
        
        differenceHamilton = hamilton - ( (numpy.sum(momentum ** 2.0) / 2.0 - energy_function(current_parameter_condinate)) )

        if random.random() < numpy.exp(differenceHamilton):
            current_parameter = current_parameter_condinate

        
        if i % 1000 == 0:
            print "%d"%(i)
            print "{0}".format(softmax(current_parameter))
            print "likelyfood:%f"%(energy_function(current_parameter_condinate))
        if i > bear_in and i % 10 == 0:
            r.append(softmax(current_parameter))
    print numpy.average(r,axis=0)
    return current_parameter
    
def callPosterior(parameter):
    #print posterior([2,2,2],parameter,0,5)
    return posterior([200,200,200,200,200,200,200,200,200,200],parameter,0,100)
def callGPosterior(parameter):
    #print gPosterior([2,2,2],parameter,0,5)
    return gPosterior([200,200,200,200,200,200,200,200,200,200],parameter,0,100)
    
x = T.dvector('x')
u = T.dscalar('u')
sigma = T.dscalar('sigma')
nSample = T.dvector('nSample')
normalPdfSyntax = 1. / T.sqrt(2 * 3.14159 * sigma ** 2) * T.exp(-(x-u) ** 2 / (2 * sigma ** 2))
posteriorSyntax = (T.sum(nSample * T.log(T.nnet.softmax(x))) + T.sum(T.log(normalPdfSyntax)))
posterior = theano.function(inputs=[nSample,x,u,sigma], outputs=posteriorSyntax)
#print posterior([2,2,1000],[1,2,3],0,5)

gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=x)
gPosterior = theano.function(inputs=[nSample,x,u,sigma], outputs=gPosteriorSyntax)
#print gPosterior([2,2,1000],[1,2,3],0,5)


estimated_parameter = hamiltonianMonteCarlo(numpy.array([0,100,0,0,0,0,0,0,0,0]),callPosterior,callGPosterior)

print softmax(estimated_parameter)


