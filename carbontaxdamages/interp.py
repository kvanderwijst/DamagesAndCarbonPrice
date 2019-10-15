import numpy as np
import numba as nb

f4 = nb.float32
f8 = nb.float64
i4 = nb.int32
i8 = nb.int64

f8_1d = nb.typeof(np.array([1.0]))
f8_2d = nb.typeof(np.array([[1.0]]))
f8_3d = nb.typeof(np.array([[[1.0]]]))


##########################
##########################
## Bilinear interpolation (2D)
##########################
##########################

@nb.njit(f8(f8,f8,  f8,f8,f8,i8,  f8,f8,f8,i8,  f8_2d), fastmath=True)
def bilinear_interpolate(x,y,  xmin,dx,xfactor,n_x,  ymin,dy,yfactor,n_y,  im):

    x0_i = np.floor((x - xmin) * xfactor)
    x1_i = x0_i + 1

    y0_i = np.floor((y - ymin) * yfactor)
    y1_i = y0_i + 1

    x0_i = int(np.maximum(0, np.minimum(n_x-1, x0_i)))
    x1_i = int(np.maximum(0, np.minimum(n_x-1, x1_i)))
    y0_i = int(np.maximum(0, np.minimum(n_y-1, y0_i)))
    y1_i = int(np.maximum(0, np.minimum(n_y-1, y1_i)))

    x0 = xmin + x0_i * dx
    x1 = xmin + x1_i * dx
    y0 = ymin + y0_i * dy
    y1 = ymin + y1_i * dy

    Ia = im[y0_i, x0_i]
    Ib = im[y1_i, x0_i]
    Ic = im[y0_i, x1_i]
    Id = im[y1_i, x1_i]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if x0_i == x1_i and y0_i == y1_i:
        return Ia
    if x0_i == x1_i:
        return Ia + (y-y0) * (Id-Ia) / (y1 - y0)
    if y0_i == y1_i:
        return Ia + (x-x0) * (Id-Ia) / (x1 - x0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id) / ((x1-x0) * (y1-y0))


##########################
##########################
## Trilinear interpolation (3D)
##########################
##########################

@nb.njit(f8(f8,f8,f8,f8,f8,f8,i8,f8,f8,f8,i8,f8,f8,f8,i8,f8_3d))
def trilinear_interpolate(x,y,z, xmin,dx,xfactor,n_x, ymin,dy,yfactor,n_y, zmin,dz,zfactor,n_z, im):

    x0_i = np.floor((x-xmin)*xfactor)
    x1_i = x0_i + 1

    y0_i = np.floor((y-ymin)*yfactor)
    y1_i = y0_i + 1

    z0_i = np.floor((z-zmin)*zfactor)
    z1_i = z0_i + 1

    x0_i = int(np.maximum(0, np.minimum(n_x-1, x0_i)))
    x1_i = int(np.maximum(0, np.minimum(n_x-1, x1_i)))
    y0_i = int(np.maximum(0, np.minimum(n_y-1, y0_i)))
    y1_i = int(np.maximum(0, np.minimum(n_y-1, y1_i)))
    z0_i = int(np.maximum(0, np.minimum(n_z-1, z0_i)))
    z1_i = int(np.maximum(0, np.minimum(n_z-1, z1_i)))

    x0 = xmin + x0_i * dx
    x1 = xmin + x1_i * dx
    y0 = ymin + y0_i * dy
    y1 = ymin + y1_i * dy
    z0 = zmin + z0_i * dz
    z1 = zmin + z1_i * dz

    c000 = im[x0_i, y0_i, z0_i]
    c001 = im[x0_i, y0_i, z1_i]
    c010 = im[x0_i, y1_i, z0_i]
    c011 = im[x0_i, y1_i, z1_i]
    c100 = im[x1_i, y0_i, z0_i]
    c101 = im[x1_i, y0_i, z1_i]
    c110 = im[x1_i, y1_i, z0_i]
    c111 = im[x1_i, y1_i, z1_i]

    # Interpolation along x-axis
    if x0_i == x1_i:
        c00, c01, c10, c11 = c000, c001, c010, c011
    else:
        xd = (x - x0) / (x1 - x0)
        c00 = c000 * (1-xd) + c100 * xd
        c01 = c001 * (1-xd) + c101 * xd
        c10 = c010 * (1-xd) + c110 * xd
        c11 = c011 * (1-xd) + c111 * xd

    # Interpolation along y-axis
    if y0_i == y1_i:
        c0, c1 = c00, c01
    else:
        yd = (y - y0) / (y1 - y0)
        c0 = c00 * (1-yd) + c10 * yd
        c1 = c01 * (1-yd) + c11 * yd

    # Interpolation along z-axis
    if z0_i == z1_i:
        c = c0
    else:
        zd = (z - z0) / (z1 - z0)
        c = c0 * (1-zd) + c1 * zd

    return c
