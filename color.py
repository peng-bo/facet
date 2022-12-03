import taichi as ti
from taichi.math import exp, clamp, mat3, inf


@ti.func
def BlackBody(t, w_nm):
    """Planck's law https://en.wikipedia.org/wiki/Planck%27s_law"""
    h = 6.6e-34  #Planck constant
    k = 1.4e-23  #Boltzmann constant
    c = 3e8  #Speed of light
    w = w_nm / 1e9
    w5 = w * w * w * w * w
    o = 0 if t < 1000 else 2 * h * (c * c) / (w5 * (exp(h * c /
                                                        (w * k * t)) - 1.0))
    return o


@ti.func
def xFit_1931(wave):
    t1 = (wave - 442.0) * (0.0624 if wave < 442.0 else 0.0374)
    t2 = (wave - 599.8) * (0.0264 if wave < 599.8 else 0.0323)
    t3 = (wave - 501.1) * (0.0490 if wave < 501.1 else 0.0382)
    return 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(
        -0.5 * t2 * t2) - 0.065 * exp(-0.5 * t3 * t3)


@ti.func
def yFit_1931(wave):
    t1 = (wave - 568.8) * (0.0213 if wave < 568.8 else 0.0247)
    t2 = (wave - 530.9) * (0.0613 if wave < 530.9 else 0.0322)
    return 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2)


@ti.func
def zFit_1931(wave):
    t1 = (wave - 437.0) * (0.0845 if wave < 437.0 else 0.0278)
    t2 = (wave - 459.0) * (0.0385 if wave < 459.0 else 0.0725)
    return 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2)


@ti.func
def xyzToLinearRgb(XYZ):
    return clamp(
        mat3([[3.240479, -1.537150, -0.498535], [
            -0.969256, 1.875991, 0.041556
        ], [0.055648, -0.204043, 1.057311]]) @ XYZ, 0.0, inf)


@ti.func
def ACESToneMapping(color):
    A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14
    color *= 0.3
    return (color * (A * color + B)) / (color * (C * color + D) + E)


@ti.func
def gamma_correction(sRGB):
    return sRGB * 12.92 if sRGB <= 0.0031308 else pow(sRGB * 1.055, 1 /
                                                      2.4) - 0.055
