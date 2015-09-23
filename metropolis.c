#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Python.h>
#include "markov_chain_monte_carlo.h"

const double PI2=2*M_PI;

double xor128(){ 
	static unsigned int x=123456789,y=362436069,z=521288629,w=88675123; 
	unsigned int t; 
	t=(x^(x<<11));
	x=y;
	y=z;
	z=w;
	w=(w^(w>>19))^(t^(t>>8));
	return (double)w/(double)0xFFFFFFFF; 
} 

double gsl_rng_uniform_pos2(){
	double r;
	do{
		r=xor128();
	}while(r==0.0);
	return r;
}

void rnorm(double *result,int n,double mean,double sd){
	double x, y, r2;
	int i;
	for(i=0;i<n;i++){
		do{
			/* choose x,y in uniform square (-1,-1) to (+1,+1) */
			x = -1 + 2 * gsl_rng_uniform_pos2 ();
			y = -1 + 2 * gsl_rng_uniform_pos2 ();

			/* see if it is in the unit circle */
			r2 = x * x + y * y;
		}while (r2 > 1.0 || r2 == 0);

		/* Box-Muller transform */
		result[i]=sd * y * sqrt (-2.0 * log (r2) / r2);
	}
}

void metropolis(double (*log_fun)(void *,double *),void *arg,double *theta,int numTheta,int mcmc){
	int iter,i;
	double *theta_can=malloc(sizeof(double)*numTheta),rnd;
	double userfun_cur=0.0;
	double t=0.0001;
	for (iter = 0; iter < mcmc; ++iter) {
		for (i = 0; i < numTheta; ++i) {
			rnorm(&rnd,1,0.0,1.0);
			theta_can[i] = theta[i]+rnd*t;	
    	}
		double userfun_can = log_fun(arg,theta_can);
		if(iter==0) userfun_cur=userfun_can-1e-10;
		const double ratio = exp(userfun_can - userfun_cur);
		if (xor128() < ratio) {
			for (i = 0; i < numTheta; ++i) {
				theta[i] = theta_can[i];
			}
			userfun_cur = userfun_can;
		}
		if ( iter > 4000000) t=0.00001;
		
	}
}

static PyObject *metropolisInterface(PyObject *self, PyObject *args){
    int nParameter,nSample;
    PyObject *sampleObj,*row;
    
    int i;
    
    if (! PyArg_ParseTuple( args, "Oi", &sampleObj, &nParameter)) return NULL;
    
    if((nSample = PyList_Size(sampleObj)) < 0) return NULL;
    int *sample=malloc(sizeof(int)*nSample);
    
    for (i=0; i<nSample; i++){
        row = PyList_GetItem(sampleObj, i);
        sample[i] = PyInt_AsLong(row);
    }
    
    double *theta=calloc(1,sizeof(double)*nParameter);
    
    MultinomialLogit *ctx = multinomialLogitInit(sample,nSample,nParameter);
    
    metropolis(multinomialLogit,ctx,theta,ctx->nParameter,600000);
    double *estimateProbability=malloc(sizeof(double)*ctx->nParameter);
    softmax(theta,estimateProbability,ctx->nParameter);


	PyListObject *list = (PyListObject *) PyList_New(ctx->nParameter);

    for(i=0;i<ctx->nParameter;i++){
         PyList_SET_ITEM(list,i,Py_BuildValue("f",estimateProbability[i]));
    }
    return Py_BuildValue("O", list);
}

static PyMethodDef mcmcmethods[] = {
    {"metropolis", metropolisInterface, METH_VARARGS},
    {NULL},
};

void initmcmc(void)
{
    Py_InitModule("mcmc", mcmcmethods);
}

