def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    # Write code here
    
    TP=TN=FP=FN = 0
    for cls in set(y_true):
        for a,y in zip(y_true,y_pred):
            if a == y == cls:
                TP+=1
            elif a != cls and y == cls:
                FP+=1
            elif a == cls and y != cls:
                FN+=1
    F1 = (2 * TP) /((2 * TP) + FP + FN)
    return F1