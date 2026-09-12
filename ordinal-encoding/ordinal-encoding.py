def ordinal_encoding(values: list, ordering: list) -> list:
    """
    Returns the ordinal index of every input value.
    """
    # Write code here
    orders = {o:i for i,o in enumerate(ordering)}
    return [orders[i] for i in values]