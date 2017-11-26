import matplotlib.pyplot as plt
import numpy as np
from miniutils import pragma
from numba import cuda, float32
from .variations import all_variations


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


class SimpleTransform:
    def __init__(self):
        self.linear = np.random.normal(size=(2, 3))

    def make_device_function(self):
        (a, b, c), (d, e, f) = self.linear.astype('float32')

        @cuda.jit(device=True)
        def device_transform(xy):
            x = xy[0]
            y = xy[1]
            xy[0] = a * x + b * y + c
            xy[1] = d * x + e * y + f

        return device_transform


class Transform:
    def __init__(self):
        self.linear = np.random.normal(size=(2, 3))
        self.nonlinear_weights = softmax(np.random.normal(size=len(all_variations)))
        self.variations = [var(*list(np.random.uniform(size=var.kind.num_p))) for var in all_variations.values()]
        self.color = softmax(np.random.uniform(size=3))
        self.post_transform = SimpleTransform()

    def make_device_function(self):
        (a, b, c), (d, e, f) = self.linear.astype('float32')
        post_transform_kernel = self.post_transform.make_device_function()
        nonlinear_weights = self.nonlinear_weights.astype('float32')
        variations = tuple(self.variations)
        num_vars = len(self.variations)

        @cuda.jit(device=True)
        @pragma.deindex(nonlinear_weights, 'nonlinear_weights')
        @pragma.deindex(variations, 'variations')
        @pragma.unroll(lv=len(variations))
        def apply_variations(xy, variation_cache, lin_x, lin_y):
            for var in range(lv):
                variations[var](variation_cache, lin_x, lin_y, a, b, c, d, e, f)
                xy[0] += nonlinear_weights[var] * variation_cache[0]
                xy[1] += nonlinear_weights[var] * variation_cache[1]

        @cuda.jit(device=True)
        def device_transform(xy):
            variation_cache = cuda.local.array(2, float32)
            lin_x = a * xy[0] + b * xy[1] + c
            lin_y = d * xy[0] + e * xy[1] + f
            xy[:] = 0.0
            variation_cache[:] = 0.0
            # variations[j](variation_cache, lin_x, lin_y, a, b, c, d, e, f)
            apply_variations(xy, variation_cache, lin_x, lin_y)
            post_transform_kernel(xy)

        return device_transform


def plot_transform(trans, xmn=-1, xmx=1, xrs=41, ymn=-1, ymx=1, yrs=41):
    trans = trans.make_device_function()

    @cuda.jit
    def check(an_array):
        thread_id = cuda.grid(1)
        pt = cuda.local.array(2, float32)
        pt[0] = an_array[thread_id, 0]
        pt[1] = an_array[thread_id, 1]
        trans(pt)
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
