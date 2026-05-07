import numpy as np


def morloc_zeros2(d1, d2):
    return np.zeros((d1, d2))

def morloc_zeros3(d1, d2, d3):
    return np.zeros((d1, d2, d3))

def morloc_zeros4(d1, d2, d3, d4):
    return np.zeros((d1, d2, d3, d4))

def morloc_zeros5(d1, d2, d3, d4, d5):
    return np.zeros((d1, d2, d3, d4, d5))


def morloc_ones2(d1, d2):
    return np.ones((d1, d2))

def morloc_ones3(d1, d2, d3):
    return np.ones((d1, d2, d3))

def morloc_ones4(d1, d2, d3, d4):
    return np.ones((d1, d2, d3, d4))

def morloc_ones5(d1, d2, d3, d4, d5):
    return np.ones((d1, d2, d3, d4, d5))


def morloc_fill2(v, d1, d2):
    return np.full((d1, d2), v)

def morloc_fill3(v, d1, d2, d3):
    return np.full((d1, d2, d3), v)

def morloc_fill4(v, d1, d2, d3, d4):
    return np.full((d1, d2, d3, d4), v)

def morloc_fill5(v, d1, d2, d3, d4, d5):
    return np.full((d1, d2, d3, d4, d5), v)


def morloc_identity(n):
    return np.eye(n)


def morloc_matmul(a, b):
    return a @ b


# Packable: tuple-of(dims, flat numpy.ndarray) <-> shaped numpy.ndarray.
# Dims are native Python ints; data is a 1D numpy array routed through the
# morloc numpy fast path (zero-copy when contiguous).
#
# Vector itself has no pack/unpack: morloc Vector maps directly to
# numpy.ndarray, so the wire form is the runtime form and no conversion
# is required.

def morloc_packMatrix(packed):
    (d1, d2), data = packed
    return np.ascontiguousarray(data).reshape((d1, d2))

def morloc_unpackMatrix(t):
    arr = np.ascontiguousarray(t)
    return ((int(arr.shape[0]), int(arr.shape[1])), arr.reshape(-1))

def morloc_packTensor3(packed):
    (d1, d2, d3), data = packed
    return np.ascontiguousarray(data).reshape((d1, d2, d3))

def morloc_unpackTensor3(t):
    arr = np.ascontiguousarray(t)
    return ((int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])),
            arr.reshape(-1))

def morloc_packTensor4(packed):
    (d1, d2, d3, d4), data = packed
    return np.ascontiguousarray(data).reshape((d1, d2, d3, d4))

def morloc_unpackTensor4(t):
    arr = np.ascontiguousarray(t)
    return ((int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])),
            arr.reshape(-1))

def morloc_packTensor5(packed):
    (d1, d2, d3, d4, d5), data = packed
    return np.ascontiguousarray(data).reshape((d1, d2, d3, d4, d5))

def morloc_unpackTensor5(t):
    arr = np.ascontiguousarray(t)
    return ((int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])),
            arr.reshape(-1))
