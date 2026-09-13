import math
def binning(values: list, num_bins: int) -> list:
    """
    Returns the equal-width bin index of every value.
    """
    # Write code here
    
    maxi = max(values)
    mini = min(values)
    if maxi == mini:
        return [0] * len(values)
    w = (maxi - mini)/num_bins
    
    def bin(v):
        return math.floor(min(((v- mini)/w),num_bins -1))
    return [bin(v) for v in values]