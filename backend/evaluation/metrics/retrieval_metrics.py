import re
from typing import List

def normalize_title(title: str) -> str:
    """
    Standardizes movie titles for robust matching by removing years,
    punctuation, extra spaces, and converting to lowercase.
    e.g., 'Interstellar (2014)' -> 'interstellar'
    e.g., 'Se7en' -> 'se7en'
    e.g., 'Toy Story 3' -> 'toy story 3'
    """
    if not title:
        return ""
    cleaned = title.lower().strip()
    # Remove year in parentheses e.g. (1994) or (2020)
    cleaned = re.sub(r'\s*\(\d{4}\)', '', cleaned)
    # Remove punctuation but preserve alphanumeric characters and spaces
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def compute_recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """
    Recall@K is the fraction of expected movies retrieved in the top K.
    """
    if not expected:
        return 1.0
    
    retrieved_k = retrieved[:k]
    retrieved_norm = {normalize_title(t) for t in retrieved_k if t}
    expected_norm = {normalize_title(t) for t in expected if t}
    
    # Check for intersection
    hits = len(retrieved_norm.intersection(expected_norm))
    return hits / len(expected_norm)

def compute_hit_rate(retrieved: List[str], expected: List[str], k: int = 10) -> float:
    """
    Hit Rate is 1.0 if at least one expected movie is in the top K retrieved, else 0.0.
    """
    if not expected:
        return 1.0
    
    retrieved_k = retrieved[:k]
    retrieved_norm = {normalize_title(t) for t in retrieved_k if t}
    expected_norm = {normalize_title(t) for t in expected if t}
    
    has_hit = len(retrieved_norm.intersection(expected_norm)) > 0
    return 1.0 if has_hit else 0.0

def compute_reciprocal_rank(retrieved: List[str], expected: List[str], k: int = 10) -> float:
    """
    Calculates the Reciprocal Rank (RR) of the first expected movie retrieved within top K.
    """
    if not expected:
        return 1.0
        
    expected_norm = {normalize_title(t) for t in expected if t}
    retrieved_k = retrieved[:k]
    
    for idx, item in enumerate(retrieved_k):
        if normalize_title(item) in expected_norm:
            return 1.0 / (idx + 1)
            
    return 0.0

def compute_retrieval_accuracy(retrieved: List[str], expected: List[str], must_not_retrieve: List[str], k: int = 10) -> float:
    """
    Retrieval Accuracy rewards retrieving expected movies and penalizes retrieving 'must_not_retrieve' movies.
    Formula: Max(0.0, (len(retrieved[:K] in expected) - len(retrieved[:K] in must_not_retrieve)) / len(expected))
    """
    if not expected:
        # If no expected movies, return 1.0 if no must_not_retrieve items were retrieved, else 0.0
        retrieved_k = retrieved[:k]
        retrieved_norm = {normalize_title(t) for t in retrieved_k if t}
        must_not_norm = {normalize_title(t) for t in must_not_retrieve if t}
        has_forbidden = len(retrieved_norm.intersection(must_not_norm)) > 0
        return 0.0 if has_forbidden else 1.0

    retrieved_k = retrieved[:k]
    retrieved_norm = {normalize_title(t) for t in retrieved_k if t}
    expected_norm = {normalize_title(t) for t in expected if t}
    must_not_norm = {normalize_title(t) for t in must_not_retrieve if t}

    correct_hits = len(retrieved_norm.intersection(expected_norm))
    forbidden_hits = len(retrieved_norm.intersection(must_not_norm))

    score = (correct_hits - forbidden_hits) / len(expected_norm)
    return max(0.0, min(1.0, score))
