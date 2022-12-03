import taichi as ti
import taichi.math as tm

Ray = ti.types.struct(origin=tm.vec3, direction=tm.vec3, wavelength=ti.i32)


@ti.func
def get_ray(u,
            v,
            wavelength,
            look_from,
            look_at=tm.vec3(0, 0, 0),
            up=tm.vec3(0, 1, 0),
            fov=25):
    d = (look_at - look_from).normalized()
    film_u = tm.cross(d, up).normalized()  #right
    film_v = tm.cross(film_u, d).normalized()  #up
    d_u = film_u * 2 * tm.tan(tm.radians(fov) / 2)
    d_v = film_v * 2 * tm.tan(tm.radians(fov) / 2)
    return Ray(look_from, (d + (u - 0.5) * d_u + (v - 0.5) * d_v).normalized(),
               wavelength)
