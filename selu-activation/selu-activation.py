import math
import numpy as np
def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    lamb = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    # Write code here
    x = np.array(x)
    return np.where(x <=0 , lamb * alpha * (np.exp(x) -1 ),lamb * x)