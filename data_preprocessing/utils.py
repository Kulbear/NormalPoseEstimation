import numpy as np
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import cv2
import pickle
import chumpy as ch
from chumpy.ch import MatVecMult
import scipy.sparse as sp
import data_preprocessing.lbs as lbs


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


"""render"""


colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.01,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam_param, cam_t, cam_rt, near=0.5, far=25, img=None):
    f = cam_param[0:2]
    c = cam_param[2:4]
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam_rt, t=cam_t, f=f, c=c)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    imtmp = simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp


"""load SMPL model"""


def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'


def ready_arguments(fname_or_dict):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='iso-8859-1')
    else:
        dd = fname_or_dict

    backwards_compatibility_replacements(dd)

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))

    return dd


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def ischumpy(x): return hasattr(x, 'dterms')


def verts_decorated(trans, pose,
                    v_template, J, weights, kintree_table, bs_style, f,
                    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):
    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = ch.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v

    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
    else:
        v_posed = v_shaped

    v = v_posed

    if sp.issparse(J):
        regressor = J
        J_tmpx = MatVecMult(regressor, v_shaped[:, 0])
        J_tmpy = MatVecMult(regressor, v_shaped[:, 1])
        J_tmpz = MatVecMult(regressor, v_shaped[:, 2])
        J = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
    else:
        assert (ischumpy(J))

    assert (bs_style == 'lbs')
    result, Jtr = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, xp=ch)

    tr = trans.reshape((1, 3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type = bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result


def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=ch):
    if xp == ch:
        assert (hasattr(pose, 'dterms'))
        assert (hasattr(v, 'dterms'))
        assert (hasattr(J, 'dterms'))
        assert (hasattr(weights, 'dterms'))

    assert (bs_style == 'lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result


def load_model(fname_or_dict):
    dd = ready_arguments(fname_or_dict)

    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }

    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1, 3))
    result.J_transformed = Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result
