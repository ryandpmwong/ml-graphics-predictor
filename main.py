# hello world
import numpy as np

def find_shadow_ellipse(light_pos, sphere_centre, sphere_radius):
    l = np.array(light_pos)
    c = np.array(sphere_centre)
    r = sphere_radius

    d = l - c
    assert d[1] > 0

    lx, ly, lz = l
    dx, dy, dz = d

    d_length = np.linalg.norm(d)

    # tan(d_pitch) = |dy| / |dx + dz|

    # cos(d_pitch) = |dy| / |d|
    d_pitch = np.arccos(dy / d_length)

    # sin(theta) = r / |d|
    theta = np.arcsin(r / d_length)

    # max_pitch = d_pitch + theta
    # min_pitch = d_pitch - theta
    max_pitch = d_pitch + theta
    min_pitch = d_pitch - theta

    # tan(max_pitch) = big_length / |ly|
    # tan(min_pitch) = lil_length / |ly|
    big_length = ly * np.tan(max_pitch)
    lil_length = ly * np.tan(min_pitch)

    # large_length = big_length - lil_length
    large_length = big_length - lil_length

    assert large_length > 0


    # sin(phi) = r / |dy|
    phi = np.arcsin(r / dy)
    # tan(phi) = half_small_length / |ly|
    # small_length = 2 * half_small_length
    small_length = 2 * ly * np.tan(phi)
    
    assert small_length > 0


    # line is l + lambda*d
    # ly + lambda * dy = 0
    mu = -ly/dy
    sx = lx + mu * dx
    sz = lz + mu * dz

    # now we need the angle
    alpha = np.arctan(dx/dz)
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    return (sx, sz, small_length, large_length, sin_alpha, cos_alpha)

def save_screen():
    # hi
    pass

def load_screen():
    pass