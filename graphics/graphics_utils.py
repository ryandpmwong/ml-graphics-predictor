from OpenGL.GL import *
import numpy as np
import matplotlib.pyplot as plt

import struct

def find_shadow_ellipse_plane_source(light_direction, sphere_centre, sphere_radius):
    """
    Finds the shape of the shadow cast onto the plane y = 0

    Assumes light_direction[1] < 0
    """
    d = np.array(light_direction)
    c = np.array(sphere_centre)
    dx, dy, dz = d
    cx, cy, cz = c
    d_length = np.linalg.norm(d)

    beta = -cy/dy
    ellipse_x = cx + beta * dx
    ellipse_z = cz + beta * dz

    minor_axis = sphere_radius
    major_axis = sphere_radius * d_length / -dy

    if dz == 0:  # alpha = 90 degrees
        sin_alpha = 1.0
        cos_alpha = 0.0
    else:
        alpha = np.arctan(dx/dz)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
    
    return (ellipse_x, ellipse_z, minor_axis, major_axis, sin_alpha, cos_alpha)


def find_shadow_ellipse_point_source(light_pos, sphere_centre, sphere_radius):
    """
    Finds the shape of the shadow cast onto the plane y = 0

    Assumes light_pos[1] > sphere_centre[1]
    """
    l = np.array(light_pos)
    c = np.array(sphere_centre)
    d = l - c
    lx, ly, lz = l
    dx, dy, dz = d
    d_length = np.linalg.norm(d)
    
    phi = np.arcsin(sphere_radius / dy)
    semiminor_axis = ly * np.tan(phi)

    d_pitch = np.arccos(dy / d_length)

    if d_pitch == 0:  # light directly above sphere centre
        semimajor_axis = semiminor_axis
        ellipse_x = float(lx)
        ellipse_z = float(lz)
    else:
        theta = np.arcsin(sphere_radius / d_length)

        # max and min pitches of light rays tangent to sphere
        max_pitch = d_pitch + theta
        min_pitch = d_pitch - theta

        large_length = ly * np.tan(max_pitch)
        small_length = ly * np.tan(min_pitch)
        semimajor_axis = (large_length - small_length) / 2

        beta = (small_length + large_length) / (2 * np.sqrt(dx ** 2 + dz ** 2))
        ellipse_x = lx - beta * dx
        ellipse_z = lz - beta * dz

    if dz == 0:  # alpha = 90 degrees
        sin_alpha = 1.0
        cos_alpha = 0.0
    else:
        alpha = np.arctan(dx/dz)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

    return (ellipse_x, ellipse_z, semiminor_axis, semimajor_axis, sin_alpha, cos_alpha)

def get_pixel_array(width, height):
    pixels = glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,None)
    screen = []
    for i in range(width*height):
        r, g, b = pixels[3*i : 3*i+3]
        grey_scale = int(0.299*r + 0.587*g + 0.114*b)
        screen.append(grey_scale)
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

def display_screen(width, height, screen):
    pixels = np.array(screen)
    pixels = pixels.reshape((height, width))
    pixels = np.flip(pixels, axis=0)
    plt.imshow(pixels, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.show()

def load_and_display_screen(filename):
    width, height, _, _, _, _, pixels = load_screen(filename)
    display_screen(width, height, pixels)