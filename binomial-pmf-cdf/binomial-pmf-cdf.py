import numpy as np
from scipy.special import comb
from scipy.stats import binom

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    return [binom.pmf(k,n,p),binom.cdf(n,p,k)]
    