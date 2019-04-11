### SICP
- 1 test funtion
``` python {.line-numbers}
### python -m doctest xx.py
###或者直接import里做
###

def gcd(m,n):
    '''Return the K largest
    k , m, n are all pso
    >>> gcd(12,8)
    4
    >>> gcd(16,12)
    4
    '''
    if m == n:
        return m
    elif m < n:
        return gcd(n,m)
    else:
        return gcd(m-n,n)

if __name__ == "__main__":
    import doctest
    doctest.testmod()



```
