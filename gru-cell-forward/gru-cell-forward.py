import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """

    def sigmoid(x):
        return 1/(1+np.exp(-x))
    # Write code here
    Wz = params["Wz"]
    Wr = params["Wr"]
    Wh = params["Wh"]
    Uz = params["Uz"]
    Ur = params["Ur"]
    Uh = params["Uh"]
    bz = params["bz"]
    br = params["br"]
    bh = params["bh"]
    


        
    zt = sigmoid(x@Wz + h_prev@Uz + bz)
    rt = sigmoid(x@Wr + h_prev@Ur + br)
    ht_ = np.tanh(
        x@Wh + (rt*h_prev) @ Uh + bh
    ) 

    ht = (1 - zt) * h_prev + zt * ht_
    return ht