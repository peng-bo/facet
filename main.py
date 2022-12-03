import numpy as np
import taichi as ti
import taichi.math as tm
from math import radians, sin, cos, tan, pi
from color import *
from ray import Ray, get_ray

ti.init(arch=ti.gpu)

ANGLES = [42.1, 41, 42.3, 35, 19.8]
PROPORTIONS = [0, 0, 0, 0, 0, 0, 0]
MAX_DEPTH = 10
HAS_LIGHT = False

plane_array = np.empty((57, 4), dtype=np.float32)
plane_field = ti.Vector.field(n=4,
                              dtype=ti.f32,
                              shape=57,
                              layout=ti.Layout.AOS)


def init():
    i = 0
    p_phi0 = radians(-ANGLES[0])
    p_phi1 = radians(-ANGLES[1])
    c_phi0 = radians(ANGLES[2])
    c_phi1 = radians(ANGLES[3])
    c_phi2 = radians(ANGLES[4])
    #hacked several meetpoints of facets
    c_r1 = sin(
        pi / 8) * (tan(c_phi0) * sin(pi / 16) - tan(c_phi1) * sin(pi / 8)) / (
            tan(c_phi0) * cos(pi / 16) - tan(c_phi1) * cos(pi / 8)) + cos(
                pi / 8)
    c_z1 = tan(c_phi0) * sin(pi / 16) * sin(pi / 8) - tan(c_phi0) * cos(
        pi / 16) * sin(pi / 8) * (tan(c_phi0) * sin(pi / 16) - tan(c_phi1) *
                                  sin(pi / 8)) / (tan(c_phi0) * cos(pi / 16) -
                                                  tan(c_phi1) * cos(pi / 8))
    c_r2 = (c_r1 * tan(c_phi2) + c_z1 -
            tan(c_phi1)) / (cos(pi / 8) * tan(c_phi2) - tan(c_phi1))
    c_z2 = (1 - c_r2) * tan(c_phi1)
    p_r1 = sin(
        pi / 8) * (tan(p_phi0) * sin(pi / 16) - tan(p_phi1) * sin(pi / 8)) / (
            tan(p_phi0) * cos(pi / 16) - tan(p_phi1) * cos(pi / 8)) + cos(
                pi / 8)
    table = c_r2 * cos(pi / 8)
    crown_height = c_z2 / 2.0
    pavilion_depth = -tan(p_phi1) / 2.0
    star_length = (c_r1 - table) / (1 - table)
    lower_half = 1 - p_r1
    PROPORTIONS[0] = crown_height + pavilion_depth  #Depth
    PROPORTIONS[1] = table
    PROPORTIONS[2] = crown_height
    PROPORTIONS[3] = ANGLES[1]
    PROPORTIONS[4] = pavilion_depth
    PROPORTIONS[5] = star_length
    PROPORTIONS[6] = lower_half
    i = cut_facet(16, 2, -2, 1.0, 0.0, p_phi0, i)  #lower half facet
    i = cut_facet(8, 4, 0, 1.0, 0.0, p_phi1, i)  #pavillion main facet
    i = cut_facet(16, 2, -2, 1.0, 0.0, c_phi0, i)  #upper half facet
    i = cut_facet(8, 4, 0, 1.0, 0.0, c_phi1, i)  #bezel facet
    i = cut_facet(8, 0, 0, c_r1, c_z1, c_phi2, i)  #star facet
    plane_array[i] = [0, 0, 1, -c_z2]  #table
    plane_field.from_numpy(plane_array)


def cut_facet(symmetry, index, index_shift, radius, height, angle, i):
    gear_index = 64
    for _ in range(symmetry):
        theta = radians(index * 360 / gear_index)
        phi = angle
        p0 = [
            cos(radians((index + index_shift) * 360 / 64)) * radius,
            sin(radians((index + index_shift) * 360 / 64)) * radius, height
        ]
        n = [
            tan(phi) * tan(phi) * cos(theta),
            tan(phi) * tan(phi) * sin(theta),
            tan(phi)
        ]
        n /= np.linalg.norm(n)
        d = -np.dot(n, p0)
        plane_array[i] = [n[0], n[1], n[2], d]
        index = index + gear_index // symmetry
        i += 1
    return i


@ti.kernel
def set_orientation():
    for i in range(57):
        plane_field[i] = plane_field[i] @ tm.rot_yaw_pitch_roll(
            0.0, 0.0, pi / 16) @ tm.rot_yaw_pitch_roll(
                0.0, -tm.radians(ANGLES[1]), 0.0)


@ti.func
def intersect(ray):
    t, t_near, t_far = tm.inf, -tm.inf, tm.inf
    index, index_near, index_far = -1, -1, -1
    eps = 1e-5
    for i in range(57):
        p = plane_field[i]
        n, d = p.xyz, p.w
        v_n = tm.dot(n, ray.origin) + d
        v_d = tm.dot(n, ray.direction)
        t = -v_n / v_d
        if v_d > 0:
            if t < t_far:
                t_far = t
                index_far = i
        elif v_d < 0:
            if t > t_near:
                t_near = t
                index_near = i

    if t_near >= t_far:
        t = tm.inf
    else:
        if eps > t_far:
            t = tm.inf
        elif eps < t_near:
            t = t_near
            index = index_near
        else:
            t = t_far
            index = index_far
    return t, index


Light = ti.types.struct(position=tm.vec3, half_angle=ti.f32)


@ti.func
def trace(ray, max_depth, has_light):
    reflectance = 1.0

    wavelength = ray.wavelength
    color_xyz = tm.vec3(0.0)
    ray_xyz = tm.vec3(xFit_1931(wavelength), yFit_1931(wavelength),
                      zFit_1931(wavelength))
    n_i = 1.0
    n_t = 2.42 + 0.044 * 1e6 / (wavelength * wavelength)
    #n_t = 2.3818 + 0.0121*1e6/(ray.wavelength*ray.wavelength)
    light = Light(tm.vec3(0, 1, 0), 0.2)  #hard coded light source
    weighted_light_return = 0.0
    depth = 0
    while depth < max_depth:
        t, index = intersect(ray)
        if t == inf:
            #Environment Lighting
            if ray.direction.z > 0:
                color_xyz = ray_xyz * reflectance
                cos_theta = tm.dot(ray.direction, tm.vec3(0, 0, 1))
                weighted_light_return += cos_theta * cos_theta * reflectance
                if has_light:
                    #Small-angle approximation
                    if tm.dot(ray.direction,
                              light.position) > (1 - light.half_angle**2):
                        color_xyz = ray_xyz * reflectance * 10  #Highlight
            break

        ray.origin += ray.direction * t

        n = plane_field[index].xyz
        nl = n if n.dot(ray.direction) < 0 else -n
        into = n.dot(nl) > 0
        eta = n_i / n_t if into else n_t / n_i

        cos_theta_i = tm.clamp(ray.direction.dot(nl), -1.0, 1.0)
        sin_theta_i = tm.sqrt(1 - cos_theta_i * cos_theta_i)
        sin_theta_t = sin_theta_i * eta
        if sin_theta_t >= 1:
            ray.direction = tm.reflect(ray.direction, nl)
        else:
            tdir = tm.refract(ray.direction, nl, eta)
            #Schlick's approximation
            a = n_t - n_i
            b = n_t + n_i
            R0 = a * a / (b * b)
            c = 1 - (-cos_theta_i if into else tdir.dot(n))
            R_eff = R0 + (1 - R0) * c * c * c * c * c

            if ti.random() < R_eff:
                reflectance *= R_eff
                ray.direction = tm.reflect(ray.direction, nl)
            else:
                reflectance *= (1 - R_eff)
                ray.direction = tm.refract(ray.direction, nl, eta)
        depth += 1
    return ray, color_xyz, weighted_light_return


image_width_dclr = image_height_dclr = 256
film_dclr = ti.Vector.field(n=3,
                            dtype=ti.f32,
                            shape=(image_width_dclr, image_height_dclr),
                            offset=(512, 0))
spp_dclr = 1024


@ti.kernel
def render_dclr():
    u, v = 0.0, 0.0
    for i in range(1000000):
        for w in range(100):
            wavelength = 380 + w * 4
            #random in disk
            theta = ti.random() * tm.pi * 2
            r = ti.random()
            x = tm.sqrt(r) * tm.cos(theta)
            y = tm.sqrt(r) * tm.sin(theta)
            #
            ray = Ray(tm.vec3(x, y, 2), tm.vec3(0, 0, -1), wavelength)
            ray_out, xyz, wlr = trace(ray, 50, False)
            if ray_out.direction.z > 0:
                r = tm.dot(ray_out.direction, tm.vec3(0, 0, 1))
                u = ray_out.direction.x * tm.degrees(
                    tm.acos(r)) * 256 / 180 + 128
                v = ray_out.direction.y * tm.degrees(
                    tm.acos(r)) * 256 / 180 + 128
            film_dclr[int(u + 0.5) + 512, int(v + 0.5)] += xyz
    for i, j in film_dclr:
        rgb = clamp(xyzToLinearRgb(film_dclr[i, j]), 0.0, inf) / 16384
        srgb = ACESToneMapping(rgb)
        film_dclr[i, j] = gamma_correction(srgb)


image_width_wlr = image_height_wlr = 256
film_wlr = ti.Vector.field(n=3,
                           dtype=ti.f32,
                           shape=(image_width_wlr, image_height_wlr),
                           offset=(512, 256))
spp_wlr = 32
swpr_wlr = 8  #sample wavelengths per ray


@ti.kernel
def render_wlr():
    sum_wlr = 0.0
    sum = 0
    I = 0.0
    for w in range(8):
        wavelength = 380.0 + (w + 0.5) * 40
        i = 1e-13 * BlackBody(6500, wavelength)
        I += i / 8
    for i, j in film_wlr:
        wlr = 0.0
        for w in range(8):
            wavelength = 380.0 + (w + 0.5) * 40
            for _ in range(spp_wlr):
                u = (i - 512 + ti.random()) / image_width_wlr
                v = (j - 256 + ti.random()) / image_height_wlr
                ray = Ray(tm.vec3(2 * u - 1, 2 * v - 1, 2), tm.vec3(0, 0, -1),
                          wavelength)
                t, __ = intersect(ray)
                if not tm.isinf(t):
                    ray_out, xyz, weighted_light_return = trace(
                        ray, 100, False)
                    wlr += weighted_light_return * 1e-13 * BlackBody(
                        6500, wavelength)
                    sum += 1
        sum_wlr += wlr

        film_wlr[i, j] = wlr / spp_wlr / swpr_wlr / I * tm.vec3(1.0)
    print("Weighted Light Return: ", sum_wlr / sum / I)


width_spectral = height_spectral = 512
spp_spectral = 32  #sample per pixel
swpr_spectral = 8  #sample wavelengths per ray
temp = ti.Vector.field(
    n=1,
    dtype=ti.f32,
    shape=(width_spectral, height_spectral, spp_spectral,
           swpr_spectral))  #Out of CUDA pre-allocated memory if n=3
film_spectral = ti.Vector.field(n=3,
                                dtype=ti.f32,
                                shape=(width_spectral, height_spectral))


@ti.kernel
def render_spectral(theta: ti.f32, max_depth: int, has_light: ti.i32):
    up = tm.vec3(0, 1, 0)
    look_from = (tm.vec4(0, 0, 10, 0) @ tm.rot_yaw_pitch_roll(
        0.0, 0.0, tm.pi / 16) @ tm.rot_yaw_pitch_roll(
            0.0, -tm.radians(ANGLES[1]), 0.0) @ tm.rot_by_axis(up, theta)).xyz
    for i, j, _, w in temp:  #loop over a dummy field
        wavelength = 380.0 + (w + ti.random()) * 50
        u = (i + ti.random()) / width_spectral
        v = (j + ti.random()) / height_spectral
        #ray = Ray(look_from,(d + (u-0.5)*d_u + (v-0.5)*d_v).normalized(),wavelength)
        ray = get_ray(u, v, wavelength, look_from=look_from)
        ray_out, xyz, wlr = trace(ray, max_depth, has_light)
        film_spectral[i, j] += xyz * 1e-13 * BlackBody(6500, ray.wavelength)

    for i, j in film_spectral:
        xyz = film_spectral[i, j] / swpr_spectral / spp_spectral
        explosure = 3
        RGB = tm.clamp(xyzToLinearRgb(xyz), 0.0, inf) * explosure
        sRGB = ACESToneMapping(RGB)
        film_spectral[i, j] = gamma_correction(sRGB)


film = ti.Vector.field(n=3, dtype=ti.f32, shape=(512 + 256, 512))


@ti.kernel
def merge_windows():
    for i, j in film:
        if i < 512:
            film[i, j] = film_spectral[i, j]
        else:
            if j < 256:
                film[i, j] = film_dclr[i, j]
            else:
                film[i, j] = film_wlr[i, j]


window = ti.ui.Window("Facet", ((512 + 256, 512)), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()


def show_options(theta):
    global ANGLES
    global MAX_DEPTH
    global HAS_LIGHT
    global WLR
    with gui.sub_window("Rotate", 0.3, 0.0, 0.3, 0.2) as w:
        w.text("Press Q/E to rotate 15 degrees")
        w.text("Press A/D to rotate 5 degrees")

    with gui.sub_window("Cut Angles", 0.00, 0.2, 0.15, 0.5) as w:
        ANGLES[0] = w.slider_float("P0", ANGLES[0], 40, 44)
        ANGLES[1] = w.slider_float("P1", ANGLES[1], 39, 43)
        ANGLES[2] = w.slider_float("C0", ANGLES[2], 40, 44)
        ANGLES[3] = w.slider_float("C1", ANGLES[3], 33, 37)
        ANGLES[4] = w.slider_float("C2", ANGLES[4], 18, 22)

        if w.button("CUT"):
            init()
            render_dclr()
            render_wlr()
            set_orientation()
            render_spectral(theta, MAX_DEPTH, HAS_LIGHT)
            merge_windows()
        if w.button("RESET"):
            ANGLES = [42.1, 41, 42.3, 35, 19.8]
            #render_wlr()
    with gui.sub_window("Render Options", 0.0, 0.0, 0.3, 0.2) as w:
        MAX_DEPTH = w.slider_int("Bounces", MAX_DEPTH, 0, 100)
        HAS_LIGHT = w.checkbox("with light source", HAS_LIGHT)
        if w.button("Save Change"):
            init()
            render_dclr()
            render_wlr()
            set_orientation()
            render_spectral(theta, MAX_DEPTH, HAS_LIGHT)
            merge_windows()
    with gui.sub_window("PROPOTIONS", 0.0, 0.7, 0.2, 0.3) as w:
        w.text("Depth: " + f'{PROPORTIONS[0]:.2%}')
        w.text("Table: " + f'{PROPORTIONS[1]:.1%}')
        w.text("Crown Height: " + f'{PROPORTIONS[2]:.2%}')
        w.text("Pavilion Angle: " + f'{PROPORTIONS[3]:.2f}')
        w.text("Pavilion Depth: " + f'{PROPORTIONS[4]:.2%}')
        w.text("Star Length: " + f'{PROPORTIONS[5]:.1%}')
        w.text("Lower Half: " f'{PROPORTIONS[6]:.1%}')


def main():
    init()
    render_dclr()
    render_wlr()
    set_orientation()
    render_spectral(0, MAX_DEPTH, HAS_LIGHT)
    merge_windows()
    theta = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'd':
                theta += pi / 360 * 5
            if window.event.key == 'a':
                theta -= pi / 360 * 5
            if window.event.key == 'e':
                theta += pi / 360 * 15
            if window.event.key == 'q':
                theta -= pi / 360 * 15
            render_spectral(theta, MAX_DEPTH, HAS_LIGHT)
            merge_windows()
        show_options(theta)
        canvas.set_image(film)
        window.show()


main()
