#include "../include/my_time_lib.h"

// Put here the implementation of mu_fn and sigma_fn

double arithmetic_mean(double *v, int len) {

    double mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (double)v[i];
    mu /= (double)len;

    return(mu);
}

double geometric_mean(double *v, int len) {
    
    double mu = 1.0;
    for (int i=0; i<len; i++) {
        mu *= (v[i] > 0) ? ((double)v[i]) : 1;
    }
    mu = pow(mu, 1.0 / len);
    
    return(mu);
}

double sigma_fn_sol(double *v, double mu, int len) {

    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        sigma += ((double)v[i] - mu)*((double)v[i] - mu);
    }
    sigma /= (double)len;

    return(sigma);
}

// -------------------------------------------------
