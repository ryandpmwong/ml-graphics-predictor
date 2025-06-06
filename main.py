from OpenGL.GL import *
import numpy as np
import matplotlib.pyplot as plt

import struct

'''def find_shadow_circle(light_pos, sphere_centre, sphere_radius):
    c = np.array(sphere_centre)
    sx = c[0]
    sz = c[2]

    small_length = sphere_radius
    large_length = sphere_radius

    sin_alpha = 0.0
    cos_alpha = 1.0
    return (sx, sz, small_length, large_length, sin_alpha, cos_alpha)'''

def find_shadow_ellipse_plane_source(light_direction, sphere_centre, sphere_radius):
    """
    Finds the shape of the shadow cast onto the plane y = 0
    """
    d = np.array(light_direction)
    c = np.array(sphere_centre)
    dx, dy, dz = d
    cx, cy, cz = c
    # c + ? d = 0
    # cy + ? dy = 0
    beta = -cy/dy
    sx = cx + beta * dx
    sz = cz + beta * dz


    small_length = sphere_radius
    large_length = sphere_radius * np.linalg.norm(d) / dy
    # sphere centre is intersection where line c + lambda l


    if dz == 0:
        sin_alpha = 1.0
        cos_alpha = 0.0
    else:
        alpha = np.arctan(dx/dz)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
    return (sx, sz, small_length, large_length, sin_alpha, cos_alpha)


def find_shadow_ellipse_point_source(light_pos, sphere_centre, sphere_radius):
    l = np.array(light_pos)
    c = np.array(sphere_centre)
    r = sphere_radius

    d = l - c

    lx, ly, lz = l
    dx, dy, dz = d
    assert dy > 0

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
    large_length = (big_length - lil_length) / 2

    assert large_length > 0


    # sin(phi) = r / |dy|
    phi = np.arcsin(r / dy)
    # tan(phi) = half_small_length / |ly|
    # small_length = 2 * half_small_length
    small_length = ly * np.tan(phi)
    
    assert small_length > 0


    # line is l + lambda*d
    # ly + lambda * dy = 0

    # |(dx + dz)| * beta = (lil_length + big_length) / 2
    beta = (lil_length + big_length) / (2 * np.sqrt(dx ** 2 + dz ** 2))

    sx = lx - beta * dx
    sz = lz - beta * dz

    # now we need the angle
    if dz == 0:
        sin_alpha = 1.0
        cos_alpha = 0.0
    else:
        alpha = np.arctan(dx/dz)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

    return (sx, sz, small_length, large_length, sin_alpha, cos_alpha)

def get_pixel_array(width, height):
    pixels = glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,None)
    screen = []
    for i in range(height):
        #this_row = []
        for j in range(width):
            r, g, b = pixels[3*(i*width + j)], pixels[3*(i*width + j) + 1], pixels[3*(i*width + j) + 2]
            gray_scale = int(0.299*r + 0.587*g + 0.114*b)
            screen.append(gray_scale)
        #screen.append(this_row)
    return screen

def save_screen(width, height, filename, sphere_centre, sphere_radius):
    # hi
    pixels = get_pixel_array(width, height)
    x, y, z = sphere_centre
    x = int(10000 * x)
    y = int(10000 * y)
    z = int(10000 * z)
    sphere_radius = int(10000 * sphere_radius)
    with open(filename, "xb") as bf:
        #bf.write(bytearray([width, height, x, y, z, sphere_radius]))
        bf.write(struct.pack("qqqqqq", width, height, x, y, z, sphere_radius))

        #bf.write(bytearray(pixels))
        bf.write(struct.pack("B" * len(pixels), *pixels))


def load_screen(filename):
    with open(filename, "rb") as bf:
        file_data = bf.read()
        #print(header)
        #print(len(header))
        width, height, x, y, z, sphere_radius = struct.unpack("qqqqqq", file_data[:48])
        #print(width, height, x, y, z, sphere_radius)
        pixels = struct.unpack("B" * width * height, file_data[48:])
        #print(pixels)
    x /= 10000
    y /= 10000
    z /= 10000
    sphere_radius /= 10000
    return (width, height, x, y, z, sphere_radius, pixels)

def display_screen(width, height, pixels):
    new_pixels = np.array(pixels)
    new_pixels = new_pixels.reshape((height, width))
    new_pixels = np.flip(new_pixels, axis=0)
    plt.imshow(new_pixels, cmap='gray')
    plt.show()

def load_and_display_screen(filename):
    width, height, _, _, _, _, pixels = load_screen(filename)
    display_screen(width, height, pixels)


'''def test_save_screen(width, height, filename, sphere_centre, sphere_radius):
    # hi
    pixels = [12,0,0,0,90,98,76,54,32]
    x, y, z = sphere_centre
    with open(filename, "xb") as bf:
        #bf.write(bytearray([width, height, x, y, z, sphere_radius]))
        bf.write(struct.pack("qqqqqq", width, height, x, y, z, sphere_radius))

        #bf.write(bytearray(pixels))
        bf.write(struct.pack("B" * len(pixels), *pixels))

def test():
    test_save_screen(3,3,"saved_screens/test_folder/test_file.dat", [314,159,264], 5280)
    width, height, x, y, z, sphere_radius, pixels = load_screen("saved_screens/test_folder/test_file.dat")
    display_screen(width, height, pixels)'''

#test()