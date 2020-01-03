import numpy as np


class Transform:
    def __init__(self, R=np.eye(3, dtype='float'), t=np.zeros(3, 'float'), s=np.ones(3, 'float')):
        self.R = R.copy()
        self.t = t.reshape(-1).copy()
        self.s = s.copy()

    def __mul__(self, other):
        R = np.dot(self.R, other.R)
        t = np.dot(self.R, other.t * self.s) + self.t
        if not hasattr(other, 's'):
            other.s = np.ones(3, 'float').copy()
        s = other.s.copy()
        return Transform(R, t, s)

    def inv(self):
        R = self.R.T
        t = -np.dot(self.R.T, self.t)
        return Transform(R, t)

    def transform(self, xyz):
        if not hasattr(self, 's'):
            self.s = np.ones(3, 'float').copy()
        assert xyz.shape[-1] == 3
        assert len(self.s) == 3
        return np.dot(xyz * self.s, self.R.T) + self.t

    def getmat4(self):
        M = np.eye(4)
        M[:3, :3] = self.R * self.s
        M[:3, 3] = self.t
        return M


def convert_param2tranform(param, scale=1):
    R = quat2R(param[0:4])
    t = param[4:7]
    s = scale * np.ones(3, 'float')
    return Transform(R, t, s)


def R2quat(M):
    """
    convert a 3x3 rotation matrix to a 4x1 quaternion

    :param np.ndarray M: (3,3) array, rotation matrix
    :return: quaternion
    :rtype: np.ndarrary
    """
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # symmetric matrix K
    K = np.array([[m00-m11-m22, 0.0, 0.0, 0.0],
                  [m01+m10, m11-m00-m22, 0.0, 0.0],
                  [m02+m20, m12+m21, m22-m00-m11, 0.0],
                  [m21-m12, m02-m20, m10-m01, m00+m11+m22]])
    K /= 3.0

    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    quat = V[[3, 0, 1, 2], np.argmax(w)]
    if quat[0] < 0.0:
        np.negative(quat, quat)

    return quat


def quat2R(quat):
    """
    Description
    ===========
    convert vector q to matrix R

    Parameters
    ==========
    :param quat: (4,) array

    Returns
    =======
    :return: (3,3) array
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    n = w * w + x * x + y * y + z * z
    s = 2. / np.clip(n, 1e-7, 1e7)

    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    R = np.stack([1 - (yy + zz), xy - wz, xz + wy,
                  xy + wz, 1 - (xx + zz), yz - wx,
                  xz - wy, yz + wx, 1 - (xx + yy)])

    return R.reshape((3, 3))


def uvd2xyz(uvd, param):
    """
    convert uvd coordinates to xyz
    return:
        points in xyz coordinates, shape [N, 3]
    """
    # fx, fy, cx, cy, w, h
    # 0,  1,  2,  3,  4, 5
    # z = d
    # x = (u - cx) * d / fx
    # y = (v - cy) * d / fy
    fx, fy, cx, cy, k1, k2, k3 = param
    assert uvd.shape[1] == 3
    z = uvd[:, 2]
    x = (uvd[:, 0] - cx) * uvd[:, 2] / fx
    y = (uvd[:, 1] - cy) * uvd[:, 2] / fy
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def projection(points, params):
    # points: [N, 3]
    fx, fy, cx, cy, k1, k2, k3 = params

    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    r2 = x ** 2 + y ** 2
    k = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    U = cx + fx * k * x
    V = cy + fy * k * y
    return np.stack((U, V), axis=-1)  # N x 2


def depth2pts(depth, param):
    fx, fy, cx, cy, k1, k2, k3 = param
    u, v = np.meshgrid(np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
                       np.linspace(0, depth.shape[0] - 1, depth.shape[0]))
    uv = np.stack([u, v, np.ones_like(u)], axis=2)  # [H, W, 3]
    uv[:, :, 0] = (uv[:, :, 0] - cx) / fx
    uv[:, :, 1] = (uv[:, :, 1] - cy) / fy
    lut = uv.copy()
    xyz = lut * np.expand_dims(depth, axis=2)
    return xyz


