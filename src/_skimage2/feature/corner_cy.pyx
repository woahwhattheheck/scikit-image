#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX
from libc.math cimport atan2, fabs, sqrt, hypot, copysign, cos, acos

from _skimage2._shared.fused_numerics cimport np_floats

cnp.import_array()


cdef void _symmetric_3x3_jacobi(
    double a, double b, double c, double d, double e, double f,
    double* eigenvalues,
) noexcept nogil:
    """Resolve clustered roots without an ill-conditioned cubic formula."""
    cdef double m[3][3]
    cdef double largest, delta, tangent, cosine, sine, x, y
    cdef int iteration, p, q, k
    m[0][0], m[1][1], m[2][2] = a, b, c
    m[0][1], m[1][0] = d, d
    m[1][2], m[2][1] = e, e
    m[0][2], m[2][0] = f, f
    # Maximum-pivot Jacobi on a 3 x 3 matrix converges quadratically. Keep a
    # fixed work bound; 32 rotations comfortably exceeds double precision.
    for iteration in range(32):
        p, q = 0, 1
        largest = fabs(m[0][1])
        if fabs(m[0][2]) > largest:
            p, q = 0, 2
            largest = fabs(m[0][2])
        if fabs(m[1][2]) > largest:
            p, q = 1, 2
            largest = fabs(m[1][2])
        if largest == 0:
            break
        delta = (m[q][q] - m[p][p]) * 0.5
        if delta == 0:
            tangent = 1
        else:
            tangent = m[p][q] / (delta + copysign(hypot(delta, m[p][q]), delta))
        cosine = 1 / sqrt(1 + tangent * tangent)
        sine = tangent * cosine
        x = tangent * m[p][q]
        m[p][p] -= x
        m[q][q] += x
        m[p][q], m[q][p] = 0, 0
        for k in range(3):
            if k != p and k != q:
                x, y = m[k][p], m[k][q]
                m[k][p] = cosine * x - sine * y
                m[p][k] = m[k][p]
                m[k][q] = sine * x + cosine * y
                m[q][k] = m[k][q]
    eigenvalues[0] = m[0][0]
    eigenvalues[1] = m[1][1]
    eigenvalues[2] = m[2][2]


def _symmetric_eigenvalues_3d(const np_floats[:, ::1] components):
    """Eigenvalues of a finite real 3D tensor batch, in ascending order.

    The six columns contain M00, M01, M02, M11, M12, M22. Scaling prevents
    overflow in the cubic invariants. Centering before squaring avoids loss
    of the traceless part near a multiple of the identity. Near repeated or
    zero roots, Jacobi rotations replace the sensitive trigonometric solve.
    """
    cdef Py_ssize_t i, n = components.shape[0]
    cdef double a, b, c, d, e, f, scale, mean, x, y, z, p, r, phi, tmp
    cdef double eigenvalues[3]
    cdef int j
    if components.shape[1] != 6:
        raise ValueError("Expected six upper-triangle tensor components.")
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    cdef np_floats[:, ::1] output = np.empty((n, 3), dtype=dtype)
    with nogil:
        for i in range(n):
            a, d, f = components[i, 0], components[i, 1], components[i, 2]
            b, e, c = components[i, 3], components[i, 4], components[i, 5]
            scale = max(fabs(a), fabs(b), fabs(c), fabs(d), fabs(e), fabs(f))
            if d == 0 and e == 0 and f == 0:
                eigenvalues[0], eigenvalues[1], eigenvalues[2] = a, b, c
            else:
                a /= scale
                b /= scale
                c /= scale
                d /= scale
                e /= scale
                f /= scale
                mean = (a + b + c) / 3
                x, y, z = a - mean, b - mean, c - mean
                p = sqrt((x*x + y*y + z*z + 2*(d*d + e*e + f*f)) / 6)
                if p == 0:
                    _symmetric_3x3_jacobi(a, b, c, d, e, f, eigenvalues)
                else:
                    x /= p
                    y /= p
                    z /= p
                    r = (x*y*z + 2*(d/p)*(e/p)*(f/p)
                         - x*(e/p)*(e/p) - y*(f/p)*(f/p)
                         - z*(d/p)*(d/p)) * 0.5
                    if fabs(r) > 0.999:
                        _symmetric_3x3_jacobi(a, b, c, d, e, f, eigenvalues)
                    else:
                        phi = acos(r) / 3
                        eigenvalues[2] = mean + 2*p*cos(phi)
                        eigenvalues[0] = mean + 2*p*cos(phi + 2.0943951023931954923)
                        eigenvalues[1] = 3*mean - eigenvalues[0] - eigenvalues[2]
                        if min(fabs(eigenvalues[0]), fabs(eigenvalues[1]),
                               fabs(eigenvalues[2])) < 1e-6:
                            _symmetric_3x3_jacobi(a, b, c, d, e, f, eigenvalues)
                for j in range(3):
                    eigenvalues[j] *= scale
            if eigenvalues[0] > eigenvalues[1]:
                tmp = eigenvalues[0]
                eigenvalues[0] = eigenvalues[1]
                eigenvalues[1] = tmp
            if eigenvalues[1] > eigenvalues[2]:
                tmp = eigenvalues[1]
                eigenvalues[1] = eigenvalues[2]
                eigenvalues[2] = tmp
            if eigenvalues[0] > eigenvalues[1]:
                tmp = eigenvalues[0]
                eigenvalues[0] = eigenvalues[1]
                eigenvalues[1] = tmp
            for j in range(3):
                output[i, j] = eigenvalues[j]
    return np.asarray(output)


def _corner_moravec(np_floats[:, ::1] cimage, Py_ssize_t window_size=1):
    """Compute Moravec corner measure response image.

    This is one of the simplest corner detectors and is comparatively fast but
    has several limitations (e.g. not rotation invariant).

    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, optional (default 1)
        Window size.

    Returns
    -------
    response : ndarray
        Moravec response image.

    References
    ----------
    .. [1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/moravec.htm
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from _skimage2.feature import corner_moravec
    >>> square = np.zeros([7, 7])
    >>> square[3, 3] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> corner_moravec(square).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 2, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    """

    cdef Py_ssize_t rows = cimage.shape[0]
    cdef Py_ssize_t cols = cimage.shape[1]

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[:, ::1] out = np.zeros((rows, cols), dtype=dtype)

    cdef np_floats msum, min_msum, t
    cdef Py_ssize_t r, c, br, bc, mr, mc

    with nogil:
        for r in range(2 * window_size, rows - 2 * window_size):
            for c in range(2 * window_size, cols - 2 * window_size):
                min_msum = DBL_MAX
                for br in range(r - window_size, r + window_size + 1):
                    for bc in range(c - window_size, c + window_size + 1):
                        if br != r and bc != c:
                            msum = 0
                            for mr in range(- window_size, window_size + 1):
                                for mc in range(- window_size, window_size + 1):
                                    t = cimage[r + mr, c + mc] - cimage[br + mr, bc + mc]
                                    msum += t * t
                            min_msum = min(msum, min_msum)

                out[r, c] = min_msum

    return np.asarray(out)


cdef inline np_floats _corner_fast_response(np_floats curr_pixel,
                                            np_floats* circle_intensities,
                                            signed char* bins, signed char
                                            state, char n) noexcept nogil:
    cdef char consecutive_count = 0
    cdef np_floats curr_response
    cdef Py_ssize_t l, m
    for l in range(15 + n):
        if bins[l % 16] == state:
            consecutive_count += 1
            if consecutive_count == n:
                curr_response = 0
                for m in range(16):
                    curr_response += fabs(circle_intensities[m] - curr_pixel)
                return curr_response
        else:
            consecutive_count = 0
    return 0


def _corner_fast(np_floats[:, ::1] image, signed char n, np_floats threshold):

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t i, j, k

    cdef signed char speed_sum_b, speed_sum_d
    cdef np_floats curr_pixel, ring_pixel
    cdef np_floats lower_threshold, upper_threshold
    cdef np_floats[:, ::1] corner_response = np.zeros((rows, cols),
                                                      dtype=dtype)

    cdef signed char *rp = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3,
                            -3, -2, -1]
    cdef signed char *cp = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1,
                            0, 1, 2, 3]
    cdef signed char bins[16]
    cdef np_floats circle_intensities[16]

    cdef cnp.float64_t curr_response

    with nogil:
        for i in range(3, rows - 3):
            for j in range(3, cols - 3):

                curr_pixel = image[i, j]
                lower_threshold = curr_pixel - threshold
                upper_threshold = curr_pixel + threshold

                # High speed test for n >= 12
                if n >= 12:
                    speed_sum_b = 0
                    speed_sum_d = 0
                    for k in range(0, 16, 4):
                        ring_pixel = image[i + rp[k], j + cp[k]]
                        if ring_pixel > upper_threshold:
                            speed_sum_b += 1
                        elif ring_pixel < lower_threshold:
                            speed_sum_d += 1
                    if speed_sum_d < 3 and speed_sum_b < 3:
                        continue

                for k in range(16):
                    circle_intensities[k] = image[i + rp[k], j + cp[k]]
                    if circle_intensities[k] > upper_threshold:
                        # Brighter pixel
                        bins[k] = b'b'
                    elif circle_intensities[k] < lower_threshold:
                        # Darker pixel
                        bins[k] = b'd'
                    else:
                        # Similar pixel
                        bins[k] = b's'

                # Test for bright pixels
                curr_response = _corner_fast_response[np_floats](curr_pixel,
                                                      circle_intensities, bins,
                                                      b'b', n)

                # Test for dark pixels
                if curr_response == 0:
                    curr_response = _corner_fast_response[np_floats](curr_pixel,
                                                          circle_intensities,
                                                          bins, b'd', n)

                corner_response[i, j] = curr_response

    return np.asarray(corner_response)


def _corner_orientations(np_floats[:, ::1] image, Py_ssize_t[:, :] corners,
                         mask):
    """Compute the orientation of corners.

    The orientation of corners is computed using the first order central moment
    i.e. the center of mass approach. The corner orientation is the angle of
    the vector from the corner coordinate to the intensity centroid in the
    local neighborhood around the corner calculated using first order central
    moment.

    Parameters
    ----------
    image : 2D array
        Input grayscale image.
    corners : (N, 2) array
        Corner coordinates as ``(row, col)``.
    mask : 2D array
        Mask defining the local neighborhood of the corner used for the
        calculation of the central moment.

    Returns
    -------
    orientations : (N, 1) array
        Orientations of corners in the range [-pi, pi].

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf
    .. [2] Paul L. Rosin, "Measuring Corner Properties"
          http://users.cs.cf.ac.uk/Paul.Rosin/corner2.pdf

    Examples
    --------
    >>> from _skimage2.morphology import octagon
    >>> from _skimage2.feature import (corner_fast, corner_peaks,
    ...                                corner_orientations)
    >>> square = np.zeros((12, 12))
    >>> square[3:9, 3:9] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corners = corner_peaks(corner_fast(square, 9), min_distance=1)
    >>> corners
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])
    >>> orientations = corner_orientations(square, corners, octagon(3, 2))
    >>> np.rad2deg(orientations)
    array([  45.,  135.,  -45., -135.])

    """

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    if mask.shape[0] % 2 != 1 or mask.shape[1] % 2 != 1:
        raise ValueError("Size of mask must be uneven.")

    cdef unsigned char[:, ::1] cmask = np.ascontiguousarray(mask != 0,
                                                            dtype=np.uint8)

    cdef Py_ssize_t i, r, c, r0, c0
    cdef Py_ssize_t mrows = mask.shape[0]
    cdef Py_ssize_t mcols = mask.shape[1]
    cdef Py_ssize_t mrows2 = (mrows - 1) / 2
    cdef Py_ssize_t mcols2 = (mcols - 1) / 2
    cdef np_floats[:, :] cimage = np.pad(image, (mrows2, mcols2),
                                         mode='constant',
                                         constant_values=0)
    cdef np_floats[:] orientations = np.zeros(corners.shape[0], dtype=dtype)
    cdef np_floats curr_pixel, m01, m10, m01_tmp

    with nogil:
        for i in range(corners.shape[0]):
            r0 = corners[i, 0]
            c0 = corners[i, 1]

            m01 = 0
            m10 = 0

            for r in range(mrows):
                m01_tmp = 0
                for c in range(mcols):
                    if cmask[r, c]:
                        curr_pixel = cimage[r0 + r, c0 + c]
                        m10 += curr_pixel * (c - mcols2)
                        m01_tmp += curr_pixel
                m01 += m01_tmp * (r - mrows2)

            orientations[i] = atan2(m01, m10)

    return np.asarray(orientations)
