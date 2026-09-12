from collections import defaultdict
def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    # Write code here
    di = defaultdict(int)

    for s in sentences:
        for w in s:
            di[w] +=1
    return di