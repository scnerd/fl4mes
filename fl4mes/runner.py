import numpy as np
import pragma
from numba import cuda, float32, int8, int32
from numba.cuda.random import create_xoroshiro128p_states as make_rng_states, xoroshiro128p_uniform_float32 as get_rng

from .transforms import SimpleTransform


def make_kernel(num_points, num_steps, transforms, transition_matrix, bounds, resolution, min_step=20,
                final_transform=None, final_color=None):
    """

    :param num_points: Number of points to execute in lock-step in kernel
    :param num_steps: Number of timesteps per point
    :param transforms: [Transform]
    :param transition_matrix: Matrix of transition probabilities from transform i to j
    :param bounds: ((xmin, xmax),(ymin,y_max))
    :param resolution: (xres, yres)
    :param min_step: The first step to actually plot a transformed point
    :returns: A cuda function that will execute the desired function
    :returns: A function that returns the output image that the cuda function writes to
    """

    ((xmin, xmax), (ymin, ymax)) = bounds
    xptp = xmax - xmin
    yptp = ymax - ymin
    (xres, yres) = resolution
    num_colors = 4
    _r = 0
    _g = 1
    _b = 2
    _a = 3
    num_transforms = len(transforms)

    # cuda_output_image = SmartArray(shape=(num_colors, yres, xres), dtype='float32', where='gpu')
    cuda_output_image = cuda.to_device(np.zeros((num_colors, yres, xres), dtype='float32'))
    transition_matrix = cuda.to_device(transition_matrix)
    return_image = cuda_output_image.copy_to_host

    transform_functions = [trans.make_device_function() for trans in transforms]
    transform_colors = cuda.to_device(np.array([trans.color for trans in transforms]))

    if final_transform is None:
        final_transform = SimpleTransform()
        final_transform.linear = np.array([[1, 0, 0], [0, 1, 0]], dtype='float32')
        final_transform = final_transform.make_device_function()

    if final_color is None:
        final_color = (1.0, 1.0, 1.0)
    final_r, final_g, final_b = final_color

    def runner(blocks):
        @cuda.jit(device=True)
        def pick_next_transform(current, thread_id, rng_states, transition_matrix):
            rng = get_rng(rng_states, thread_id)
            for i in range(num_transforms):
                if rng <= transition_matrix[current, i]:
                    return i
                rng -= transition_matrix[current, i]
            return 0

        @cuda.jit(device=True)
        def hist(val, vmin, vptp, vres):
            return int32(((val - vmin) / vptp) * vres)

        @cuda.jit(device=True)
        @pragma.deindex(transform_functions, 'transform_functions')
        @pragma.unroll()
        def call_transform(i, pt):
            for j in range(len(transform_functions)):
                if i == j:
                    transform_functions[j](pt)

        @cuda.jit
        def kernel(transform_colors, rng_states, transition_matrix, out):
            thread_id = cuda.grid(1)
            # if thread_id == 0:
            #    from pdb import set_trace; set_trace()

            transform_ids = cuda.shared.array(num_steps, dtype=int8)
            transform_ids[0] = 0
            if cuda.threadIdx.x == 0:
                for i in range(1, num_steps):
                    transform_ids[i] = pick_next_transform(transform_ids[i - 1], thread_id, rng_states,
                                                           transition_matrix)
            cuda.syncthreads()

            pt = cuda.local.array(2, float32)
            pt[0] = get_rng(rng_states, thread_id) * 2 - 1
            pt[1] = get_rng(rng_states, thread_id) * 2 - 1
            r = g = b = 1.0
            final_pt = cuda.local.array(2, float32)

            for i in range(num_steps):
                call_transform(i, pt)
                r = (r + transform_colors[i, _r]) / 2
                g = (g + transform_colors[i, _g]) / 2
                b = (b + transform_colors[i, _b]) / 2

                if i >= min_step:
                    final_pt[0] = pt[0]
                    final_pt[1] = pt[1]
                    final_transform(final_pt)
                    rf = (r + final_r) / 2
                    gf = (g + final_g) / 2
                    bf = (b + final_b) / 2

                    xbin = hist(final_pt[0], xmin, xptp, xres)
                    ybin = hist(final_pt[1], ymin, yptp, yres)
                    cuda.atomic.add(out, (_r, ybin, xbin), rf)
                    cuda.atomic.add(out, (_g, ybin, xbin), gf)
                    cuda.atomic.add(out, (_b, ybin, xbin), bf)
                    cuda.atomic.add(out, (_a, ybin, xbin), 1.0)

        threads_per_block = num_points
        rng_states = make_rng_states(threads_per_block * blocks, seed=0)
        kernel[(blocks,), threads_per_block](transform_colors, rng_states, transition_matrix, cuda_output_image)

    return runner, return_image
