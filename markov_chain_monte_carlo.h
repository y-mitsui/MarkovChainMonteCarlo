typedef struct{
	int nParameter;
	int *aggregate;
}MultinomialLogit;

double lnorm(double x,double mean,double var);
void softmax(double *parameter,double *result,int nParameter);
double multinomialLogit(void *arg,double *parameter);
MultinomialLogit* multinomialLogitInit(int *sample,int nSample,int nParameter);

