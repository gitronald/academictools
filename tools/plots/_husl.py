"""HUSL color space conversions, ported from husl-python 2.1.0 as seaborn vendors it.

Only the two conversions the ``dark:`` and ``light:`` palettes need are kept:
``rgb_to_husl`` and ``husl_to_rgb``. The arithmetic is unchanged, so the
palettes match seaborn's exactly.

Copyright (c) 2012 Alexei Boronine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

Triple = tuple[float, float, float]

_M = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)
_M_INV = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)

# D65 illuminant
_REF_Y = 1.00000
_REF_U = 0.19784
_REF_V = 0.46834
_LAB_E = 0.008856
_LAB_K = 903.3


def husl_to_rgb(h: float, s: float, l: float) -> Triple:  # noqa: E741
    """Convert HUSL (hue in degrees, saturation and lightness 0-100) to RGB."""
    return _lch_to_rgb(*_husl_to_lch(h, s, l))


def rgb_to_husl(r: float, g: float, b: float) -> Triple:
    """Convert RGB (channels 0-1) to HUSL."""
    return _lch_to_husl(*_rgb_to_lch(r, g, b))


def _lch_to_rgb(l: float, c: float, h: float) -> Triple:  # noqa: E741
    return _xyz_to_rgb(*_luv_to_xyz(*_lch_to_luv(l, c, h)))


def _rgb_to_lch(r: float, g: float, b: float) -> Triple:
    return _luv_to_lch(*_xyz_to_luv(*_rgb_to_xyz(r, g, b)))


def _max_chroma(L: float, H: float) -> float:
    hrad = math.radians(H)
    sin_h = math.sin(hrad)
    cos_h = math.cos(hrad)
    sub1 = math.pow(L + 16, 3.0) / 1560896.0
    sub2 = sub1 if sub1 > 0.008856 else (L / 903.3)
    result = float("inf")
    for m1, m2, m3 in _M:
        top = (0.99915 * m1 + 1.05122 * m2 + 1.14460 * m3) * sub2
        rbottom = 0.86330 * m3 - 0.17266 * m2
        lbottom = 0.12949 * m3 - 0.38848 * m1
        bottom = (rbottom * sin_h + lbottom * cos_h) * sub2
        for t in (0.0, 1.0):
            C = L * (top - 1.05122 * t) / (bottom + 0.17266 * sin_h * t)
            if 0.0 < C < result:
                result = C
    return result


def _dot(a: Triple, b: Triple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _f(t: float) -> float:
    if t > _LAB_E:
        return math.pow(t, 1.0 / 3.0)
    return 7.787 * t + 16.0 / 116.0


def _f_inv(t: float) -> float:
    if math.pow(t, 3.0) > _LAB_E:
        return math.pow(t, 3.0)
    return (116.0 * t - 16.0) / _LAB_K


def _from_linear(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * math.pow(c, 1.0 / 2.4) - 0.055


def _to_linear(c: float) -> float:
    a = 0.055
    if c > 0.04045:
        return math.pow((c + a) / (1.0 + a), 2.4)
    return c / 12.92


def _xyz_to_rgb(x: float, y: float, z: float) -> Triple:
    xyz = (x, y, z)
    r, g, b = (_from_linear(_dot(row, xyz)) for row in _M)
    return (r, g, b)


def _rgb_to_xyz(r: float, g: float, b: float) -> Triple:
    rgbl = (_to_linear(r), _to_linear(g), _to_linear(b))
    x, y, z = (_dot(row, rgbl) for row in _M_INV)
    return (x, y, z)


def _xyz_to_luv(X: float, Y: float, Z: float) -> Triple:
    if X == Y == Z == 0.0:
        return (0.0, 0.0, 0.0)
    var_u = (4.0 * X) / (X + (15.0 * Y) + (3.0 * Z))
    var_v = (9.0 * Y) / (X + (15.0 * Y) + (3.0 * Z))
    L = 116.0 * _f(Y / _REF_Y) - 16.0
    if L == 0.0:
        return (0.0, 0.0, 0.0)
    U = 13.0 * L * (var_u - _REF_U)
    V = 13.0 * L * (var_v - _REF_V)
    return (L, U, V)


def _luv_to_xyz(L: float, U: float, V: float) -> Triple:
    if L == 0:
        return (0.0, 0.0, 0.0)
    var_y = _f_inv((L + 16.0) / 116.0)
    var_u = U / (13.0 * L) + _REF_U
    var_v = V / (13.0 * L) + _REF_V
    Y = var_y * _REF_Y
    X = 0.0 - (9.0 * Y * var_u) / ((var_u - 4.0) * var_v - var_u * var_v)
    Z = (9.0 * Y - (15.0 * var_v * Y) - (var_v * X)) / (3.0 * var_v)
    return (X, Y, Z)


def _luv_to_lch(L: float, U: float, V: float) -> Triple:
    C = math.pow(math.pow(U, 2) + math.pow(V, 2), 1.0 / 2.0)
    H = math.degrees(math.atan2(V, U))
    if H < 0.0:
        H = 360.0 + H
    return (L, C, H)


def _lch_to_luv(L: float, C: float, H: float) -> Triple:
    hrad = math.radians(H)
    return (L, math.cos(hrad) * C, math.sin(hrad) * C)


def _husl_to_lch(H: float, S: float, L: float) -> Triple:
    if L > 99.9999999:
        return (100, 0.0, H)
    if L < 0.00000001:
        return (0.0, 0.0, H)
    return (L, _max_chroma(L, H) / 100.0 * S, H)


def _lch_to_husl(L: float, C: float, H: float) -> Triple:
    if L > 99.9999999:
        return (H, 0.0, 100.0)
    if L < 0.00000001:
        return (H, 0.0, 0.0)
    return (H, C / _max_chroma(L, H) * 100.0, L)
