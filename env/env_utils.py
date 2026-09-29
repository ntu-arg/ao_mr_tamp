import numpy as np

def get_subset(set, k):
    '''
        Get all subsets of size k from the given set.
    '''
    if k == 0:
        return [[]]
    if not set:
        return []
    
    first = set[0]
    rest = set[1:]

    subsets_with_first = [[first] + subset for subset in get_subset(rest, k-1)]
    subsets_without_first = get_subset(rest, k)

    return subsets_with_first + subsets_without_first

def get_all_subsets(set):
    '''
        Get all subsets of the given set.
    '''
    all_subsets = []
    for k in range(1, len(set) + 1):
        all_subsets.extend(get_subset(set, k))
    return all_subsets


def interpolate_motion(q1, q2, t):
    """
    Linearly interpolate one or more 7-joint robot configurations.

    ``q1`` and ``q2`` may have shape ``(7,)`` for one robot or
    ``(n_robots, 7)`` for multiple robots. The returned list preserves that
    input shape.
    """
    q1_array = np.asarray(q1, dtype=float)
    q2_array = np.asarray(q2, dtype=float)

    if q1_array.shape != q2_array.shape:
        raise ValueError("q1 and q2 must have the same shape")
    if q1_array.ndim not in (1, 2) or q1_array.shape[-1] != 7:
        raise ValueError("configurations must have shape (7,) or (n_robots, 7)")

    return (q1_array + t * (q2_array - q1_array)).tolist()
