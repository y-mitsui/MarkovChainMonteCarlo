#include "markov_chain_monte_carlo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const double PI2=2*M_PI;

double lnorm(double x,double mean,double var){
    return log(1.0/sqrt(PI2*var*var)*exp(-(x-mean)*(x-mean)/(2*var*var)));
}

void softmax(double *parameter,double *result,int nParameter){
    double sum = 0.0;
    int i;
    for(i=0;i<nParameter;i++){
        sum += exp(parameter[i]);
    }
    for(i=0;i<nParameter;i++){
        result[i]=exp(parameter[i])/sum;
    }
}

double multinomialLogit(void *arg,double *parameter){
    MultinomialLogit *likeArg=arg;
    double *prob=malloc(sizeof(double)*likeArg->nParameter);
    double r = 0.0;
    int i;
    for(i=0;i<likeArg->nParameter;i++){
        r += lnorm(parameter[i],0,5);
    }
    softmax(parameter,prob,likeArg->nParameter);
    for(i=0;i<likeArg->nParameter;i++){
        r += likeArg->aggregate[i] * log(prob[i]);
    }
    return r;
}

MultinomialLogit* multinomialLogitInit(int *sample,int nSample,int nParameter){
    int i;
    MultinomialLogit* r=malloc(sizeof(MultinomialLogit));
    r->nParameter=nParameter;
    r->aggregate=calloc(1,sizeof(int)*nParameter);
    for(i=0;i<nSample;i++){
            r->aggregate[sample[i]]++;
    }

    return r;
}
