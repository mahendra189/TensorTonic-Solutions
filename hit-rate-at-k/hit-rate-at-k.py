def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    
    return sum([1 for i in range(len(recommendations)) if len(list(set(recommendations[i][:k]) & set(ground_truth[i]))) > 0 ]) / len(recommendations)
        
        
    