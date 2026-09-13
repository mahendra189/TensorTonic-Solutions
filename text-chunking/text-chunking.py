def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    # Write code here
    chunks = []
    current = 0
    while  current < len(tokens):
        chunks.append(tokens[current:current+chunk_size])
        current += chunk_size
        if current != len(tokens):
            current -= overlap
    return chunks        
            