def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    blocked = set(stopwords)
    return [ w for w in tokens if w not in blocked]