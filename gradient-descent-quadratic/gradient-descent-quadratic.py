def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    # def call(a,b,c,x0):
    #     return a*x0**2+b*x0+c
    # for s in range(steps):
    #     value = call(a,b,c,x0)
    #     x0 -= lr
    # return x0
    for _ in range(steps):
        grad = 2 * a * x0 + b   # derivative
        x0 = x0 - lr * grad     # update step
    return x0