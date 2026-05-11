import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    # Update biased first moment estimate
    m_t = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)

    # Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_t, v_t