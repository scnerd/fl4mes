import math

import numpy as np
from numba import cuda, float32

all_variations = {}


class Variation:
    def __init__(self, kind, f):
        self.kind = kind
        self.f = f
        self.func_name = f.__name__
        self.p = None

    @property
    def num_p(self):
        return self.kind.num_p

    def __call__(self, p=[]):
        assert len(p) <= self.num_p
        p = [p[i] if i < len(p) else 0 for i in range(self.num_p)]
        return cuda.jit(device=True)(self.f(p))


class VariationKind:
    def __init__(self, num_p):
        self.num_p = num_p

    def __call__(self, f):
        inner_variation_maker = Variation(self, f)
        all_variations[f.__name__] = inner_variation_maker
        return inner_variation_maker


@VariationKind(0)
def linear(p=[]):
    def f(out_xy, x, y, a, b, c, d, e, f):
        out_xy[0] = x
        out_xy[1] = y

    return f


@VariationKind(0)
def sinusoidal(p=[]):
    def f(out_xy, x, y, a, b, c, d, e, f):
        out_xy[0] = math.sin(x)
        out_xy[1] = math.sin(y)

    return f


@VariationKind(0)
def spherical(p=[]):
    def f(out_xy, x, y, a, b, c, d, e, f):
        r = math.sqrt(x ** 2 + y ** 2)
        r = 1 / r ** 2
        out_xy[0] = r * x
        out_xy[1] = r * y

    return f


@VariationKind(0)
def swirl(p=[]):
    def f(out_xy, x, y, a, b, c, d, e, f):
        r2 = x ** 2 + y ** 2
        out_xy[0] = x * math.sin(r2) - y * math.cos(r2)
        out_xy[1] = x * math.cos(r2) + y * math.sin(r2)

    return f


def plot_variation(var, xmn=-1, xmx=1, xrs=41, ymn=-1, ymx=1, yrs=41):
    from matplotlib import pyplot as plt

    @cuda.jit
    def check(an_array):
        thread_id = cuda.grid(1)
        pt = cuda.local.array(2, float32)
        pt[0] = an_array[thread_id, 0]
        pt[1] = an_array[thread_id, 1]
        var(pt, pt[0], pt[1], 1, 1, 1, 1, 1, 1)
        an_array[thread_id, 0] = pt[0]
        an_array[thread_id, 1] = pt[1]

    xs = np.linspace(xmn, xmx, xrs)
    ys = np.linspace(ymn, ymx, yrs)

    grid_xs, grid_ys = np.meshgrid(xs, ys)
    pts = np.stack([grid_xs.ravel(), grid_ys.ravel()]).T

    check[(len(pts), 1), 1](pts)
    pts = pts.reshape((xrs, yrs, 2))

    plt.figure(figsize=(6, 6))
    for y in range(yrs):
        plt.plot(pts[:, y, 0], pts[:, y, 1], c='k')
    for x in range(xrs):
        plt.plot(pts[x, :, 0], pts[x, :, 1], c='k')
    plt.show()
