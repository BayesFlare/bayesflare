import sys, os, errno

def nextpow2(i):
    """
    Calculates the nearest power of two to the inputed number.

    Parameters
    ----------
    i : int
       An integer.

    Output
    ------
    n : int
       The power of two closest to `i`.
    """
    n = 1
    while n < i: n *= 2
    return n

def mkdir(path):
    """
    Recursively makes a folder.

    Parameters
    ----------
    path : str
       A string describing the path where the folder should be created.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
