from collections import Counter,defaultdict
def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    sorted_values = sorted(values)
    c = Counter(sorted_values)
    su = defaultdict(int)
    for i,v in enumerate(sorted_values):
        su[v] += i+1

    rankings = {}
    for v in c:
        rankings[v] = su[v] / c[v]
    return [rankings[v] for v in values]
    
    