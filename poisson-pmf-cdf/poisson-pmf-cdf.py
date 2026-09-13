import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    pmf = (math.e**(-lam)*lam**(k))/math.factorial(k)
    cdf = sum((math.e**(-lam)*lam**(i))/math.factorial(i) for i in range(k+1))
    return {
        "pmf":float(pmf),
        "cdf":float(cdf)
    }