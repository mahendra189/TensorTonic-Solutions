import math
from collections import Counter
import numpy as np

def bm25_score(
    query_tokens: list[str],
    docs: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75
) -> np.ndarray:
    """
    Returns a NumPy array with one BM25 score per document.
    """

    N = len(docs)

    # Length of each document
    doc_lengths = np.array([len(doc) for doc in docs])

    # Average document length
    avgdl = np.mean(doc_lengths) if N > 0 else 0

    # Document frequency for each term
    df = Counter()

    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []

    for doc in docs:
        tf = Counter(doc)
        score = 0.0

        for term in query_tokens:

            # Term does not occur in this document
            if tf[term] == 0:
                continue

            # Number of documents containing the term
            term_df = df[term]

            # IDF
            idf = math.log(
                1 + (N - term_df + 0.5) / (term_df + 0.5)
            )

            # Term frequency
            term_tf = tf[term]

            # Document length normalization
            denominator = (
                term_tf
                + k1 * (
                    1 - b
                    + b * len(doc) / avgdl
                )
            )

            # BM25 contribution
            score += idf * (
                term_tf * (k1 + 1)
            ) / denominator

        scores.append(score)

    return np.array(scores)