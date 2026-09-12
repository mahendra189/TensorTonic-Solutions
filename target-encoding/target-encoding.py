def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    # Write code here
    means = {}
    for category in set(categories):
        values = [
            target
            for cat,target in zip(categories,targets)
            if cat == category
        ]
        means[category] = sum(values)/len(values)
    return [means[category] for category in categories]