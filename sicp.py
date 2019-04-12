def gcd(m,n):
    """Return the K largest

    k , m, n are all pso

    >>> gcd(12,8)
    4
    >>> gcd(16,12)
    4
    """
    pass
    '''
    if m == n:
        return m
    elif m < n:
        return gcd(n,m)
    else:
        return gcd(m-n,n)
    '''
##

