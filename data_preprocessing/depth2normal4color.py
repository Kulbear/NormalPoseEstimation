import sys
sys.path.append('../')
import cv2
import pickle
import os
from data_preprocessing import utils as utils
import glob
import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import multiprocessing


def connectComponent(img, uvd):
    # stats [x0, y0, width, height, area] N*5
    _, _, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8))
    stat = stats[np.argmax(stats[1:, 4]) + 1, :]
    u_min = stat[0]
    v_min = stat[1]
    u_max = stat[0] + stat[2]
    v_max = stat[1] + stat[3]

    idx = (u_min < uvd[:, 0]) & (uvd[:, 0] < u_max) & (v_min < uvd[:, 1]) & (uvd[:, 1] < v_max)
    uvd = uvd[idx, :]
    return uvd


def obtain_normal_single_processor(filenames, root_dir, params0906, params0909, cpu_i, color_cam_num=3):
    H, W = 1080, 1920
    for idx, filename in enumerate(filenames):
        # if 'guochuan_demo' not in filename:
        #     continue
        # 'd1p', 'd2p', 'd3p', 'cd1', 'cd2', 'cd3', 'param_p',
        # 'param_c1', 'param_d1', 'param_c2', 'param_d2', 'param_c3', 'param_d3'

        # load correpsonding camera params
        name = filename.split('_')[0]
        if name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting', 'guochuan']:
            param_d1 = params0906['param_d1']
            param_d2 = params0906['param_d2']
            param_d3 = params0906['param_d3']
            param_c1 = params0906['param_c1']
            param_c2 = params0906['param_c2']
            param_c3 = params0906['param_c3']

            T_d1p = utils.convert_param2tranform(params0906['d1p'])
            T_pd1 = T_d1p.inv()
            T_d2p = utils.convert_param2tranform(params0906['d2p'])
            T_pd2 = T_d2p.inv()
            T_d3p = utils.convert_param2tranform(params0906['d3p'])
            T_pd3 = T_d3p.inv()

            if color_cam_num == 1:
                param_c = param_c1
                T_cd1 = utils.convert_param2tranform(params0906['cd1'])
                T_cd2 = T_cd1 * T_d1p * T_pd2
                T_cd3 = T_cd1 * T_d1p * T_pd3
            elif color_cam_num == 2:
                param_c = param_c2
                T_cd2 = utils.convert_param2tranform(params0906['cd2'])
                T_cd1 = T_cd2 * T_d2p * T_pd1
                T_cd3 = T_cd2 * T_d2p * T_pd3
            elif color_cam_num == 3:
                param_c = param_c3
                T_cd3 = utils.convert_param2tranform(params0906['cd3'])
                T_cd1 = T_cd3 * T_d3p * T_pd1
                T_cd2 = T_cd3 * T_d3p * T_pd2
            else:
                raise ValueError('color camera num errors %i. It should be 1, 2 or 3.' % color_cam_num)

        else:
            param_d1 = params0909['param_d1']
            param_d2 = params0909['param_d2']
            param_d3 = params0909['param_d3']
            param_c1 = params0909['param_c1']
            param_c2 = params0909['param_c2']
            param_c3 = params0909['param_c3']

            T_d1p = utils.convert_param2tranform(params0909['d1p'])
            T_pd1 = T_d1p.inv()
            T_d2p = utils.convert_param2tranform(params0909['d2p'])
            T_pd2 = T_d2p.inv()
            T_d3p = utils.convert_param2tranform(params0909['d3p'])
            T_pd3 = T_d3p.inv()

            if color_cam_num == 1:
                param_c = param_c1
                T_cd1 = utils.convert_param2tranform(params0909['cd1'])
                T_cd2 = T_cd1 * T_d1p * T_pd2
                T_cd3 = T_cd1 * T_d1p * T_pd3
            elif color_cam_num == 2:
                param_c = param_c2
                T_cd2 = utils.convert_param2tranform(params0909['cd2'])
                T_cd1 = T_cd2 * T_d2p * T_pd1
                T_cd3 = T_cd2 * T_d2p * T_pd3
            elif color_cam_num == 3:
                param_c = param_c3
                T_cd3 = utils.convert_param2tranform(params0909['cd3'])
                T_cd1 = T_cd3 * T_d3p * T_pd1
                T_cd2 = T_cd3 * T_d3p * T_pd2
            else:
                raise ValueError('color camera num errors %i. It should be 1, 2 or 3.' % color_cam_num)

        save_dir = '%s/color%i_noisy_normal/%s' % (root_dir, color_cam_num, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        N = len(glob.glob('%s/annotation_openpose_kinect/fusion/%s/*.pkl' % (root_dir, filename)))
        print('working on %s (%i), %i examples, (save directory %s)' % (filename, idx, N, save_dir))
        d1_files = ['%s/depth/PC1/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
        d2_files = ['%s/depth/PC2/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
        d3_files = ['%s/depth/PC3/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]

        # c_files = ['%s/color/PC%i/%s/color_%i.jpg' % (root_dir, color_cam_num, filename, i) for i in range(N)]
        # mesh_files = ['%s/fusion_depth_mesh/%s/depth_mesh_%i.ply' % (root_dir, filename, i) for i in range(N)]
        seg1_files = ['%s/segmentation_plane_fitting/PC1/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]
        seg2_files = ['%s/segmentation_plane_fitting/PC2/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]
        seg3_files = ['%s/segmentation_plane_fitting/PC3/%s/seg_depth_%i.mat' %
                      (root_dir, filename, i) for i in range(N)]

        start_time = time.time()
        time_idx = 0
        for i in range(N):
            # if i != 1000:
            #     continue
            # print(c_files[i])
            # time1 = time.time()

            time_idx += 1
            if i % 300 == 1:
                end_time = time.time()
                print('[cpu %2d, %i-th file] %s (%i / %i), %.2f s/frame'
                      % (cpu_i, idx, filename, i, N, (end_time - start_time) / time_idx))

            if os.path.exists('%s/normal_%i.pkl' % (save_dir, i)):
                try:
                    with open('%s/normal_%i.pkl' % (save_dir, i), 'rb') as f:
                        _ = pickle.load(f)
                    continue
                except:
                    pass
            try:
                img_d1 = cv2.imread(d1_files[i], -1)[:, ::-1]
                img_d2 = cv2.imread(d2_files[i], -1)[:, ::-1]
                img_d3 = cv2.imread(d3_files[i], -1)[:, ::-1]

                seg_uv1 = loadmat(seg1_files[i])['uv'] - 1
                seg_uv2 = loadmat(seg2_files[i])['uv'] - 1
                seg_uv3 = loadmat(seg3_files[i])['uv'] - 1

                uvd1 = np.stack([seg_uv1[:, 0], seg_uv1[:, 1], img_d1[seg_uv1[:, 1], seg_uv1[:, 0]]], axis=1)
                uvd2 = np.stack([seg_uv2[:, 0], seg_uv2[:, 1], img_d2[seg_uv2[:, 1], seg_uv2[:, 0]]], axis=1)
                uvd3 = np.stack([seg_uv3[:, 0], seg_uv3[:, 1], img_d3[seg_uv3[:, 1], seg_uv3[:, 0]]], axis=1)

                img1 = np.zeros_like(img_d1)
                img1[uvd1[:, 1], uvd1[:, 0]] = uvd1[:, 2]
                img2 = np.zeros_like(img_d2)
                img2[uvd2[:, 1], uvd2[:, 0]] = uvd2[:, 2]
                img3 = np.zeros_like(img_d3)
                img3[uvd3[:, 1], uvd3[:, 0]] = uvd3[:, 2]

                uvd1 = connectComponent(img1, uvd1)
                uvd2 = connectComponent(img2, uvd2)
                uvd3 = connectComponent(img3, uvd3)

                xyz_d1 = T_cd1.transform(utils.uvd2xyz(uvd1, param_d1))
                xyz_d2 = T_cd2.transform(utils.uvd2xyz(uvd2, param_d2))
                xyz_d3 = T_cd3.transform(utils.uvd2xyz(uvd3, param_d3))
            except:
                print('[warning] %s' % seg1_files[i])
                continue

            # point_cloud1 = T_pd1.transform(np.reshape(utils.depth2pts(img_d1, param_d1), [-1, 3]))
            # point_cloud2 = T_pd2.transform(np.reshape(utils.depth2pts(img_d2, param_d2), [-1, 3]))
            # point_cloud3 = T_pd3.transform(np.reshape(utils.depth2pts(img_d3, param_d3), [-1, 3]))
            # point_cloud = np.concatenate([point_cloud1, point_cloud2, point_cloud3], axis=0)
            # point_cloud = point_cloud2

            # plydata = PlyData.read(mesh_files[i])
            # vertex = plydata['vertex'].data[:]
            # point_cloud = np.stack([np.array([vertex[ii][0], vertex[ii][1], vertex[ii][2]])
            #                         for ii in range(vertex.size)], axis=0) * 1000

            # time2 = time.time()
            results = []
            for point_cloud in [xyz_d1, xyz_d2, xyz_d3]:
                if point_cloud.shape[0] < 1000:
                    print('[warning] less than 1000 point, %s' % seg1_files[i])
                    continue

                uv = utils.projection(point_cloud, param_c).astype(np.int32)
                point_idx = (0 <= uv[:, 0]) & (uv[:, 0] < W) & (0 <= uv[:, 1]) & (uv[:, 1] < H)
                uv = uv[point_idx, :]
                d = point_cloud[point_idx, 2]
                # print(uv.shape)

                img = np.zeros([H, W])
                img[uv[:, 1], uv[:, 0]] = d
                # img = img[:, 100:1124]
                tmp = np.where(img > 0)
                u_min = max(np.min(tmp[1]) - 10, 0)
                u_max = min(np.max(tmp[1]) + 10, W)
                v_min = max(np.min(tmp[0]) - 10, 0)
                v_max = min(np.max(tmp[0]) + 10, H)
                img_crop = img[v_min: v_max, u_min: u_max]
                # print('image size', u_max-u_min, v_max-v_min)

                fx, fy, cx, cy, k1, k2, k3 = param_c
                cx = cx - u_min
                cy = cy - v_min
                new_param_p = (fx, fy, cx, cy, k1, k2, k3)

                lut_xyz = utils.depth2pts(img_crop, new_param_p)
                k = 4
                threshold = 30
                cart_normal = np.zeros([v_max - v_min, u_max - u_min, 3])
                for u in range(u_max - u_min):
                    for v in range(v_max - v_min):
                        u_idx_min = max(0, u - k)
                        u_idx_max = min(u + k + 1, W)
                        v_idx_min = max(0, v - k)
                        v_idx_max = min(v + k + 1, H)
                        points = np.reshape(lut_xyz[v_idx_min:v_idx_max, u_idx_min:u_idx_max], [-1, 3])
                        if np.sum(points[:, 2] > 0) > 5:
                            points = points[points[:, 2] > 0]
                            distances = np.sqrt(np.sum((points - np.mean(points, axis=0)) ** 2, axis=1))
                            if np.sum(np.logical_and(0 < distances, distances < threshold)) > 5:
                                pointsNN = points[np.logical_and(0 < distances, distances < threshold), :]
                                # pca = PCA(n_components=2)
                                # pca.fit(pointsNN)
                                # coeff = pca.components_
                                # query_normal = np.cross(coeff[0, :], coeff[1, :])
                                query_normal = np.sum(np.dot(np.linalg.pinv(np.dot(pointsNN.T, pointsNN)), pointsNN.T),
                                                      axis=1)
                                query_normal = query_normal / np.linalg.norm(query_normal)
                                if query_normal[2] < 0:
                                    query_normal = - query_normal
                                cart_normal[v, u, :] = np.clip(query_normal, -1, 1)

                img_result = np.zeros([H, W, 3])
                img_result[v_min: v_max, u_min: u_max] = cart_normal
                results.append(img_result)

                # tmp = (cart_normal + 1) / 2
                #
                # plt.figure(figsize=(12, 8))
                # plt.subplot(121)
                # plt.imshow(img_crop, cmap='gray')
                # plt.axis('off')
                #
                # plt.subplot(122)
                # plt.imshow(tmp)
                # plt.axis('off')
                # plt.show()
            # time3 = time.time()

            img_normal = None
            for result in results:
                if img_normal is None:
                    img_normal = result
                else:
                    img_normal = img_normal + result
            img_tmp = np.linalg.norm(img_normal, axis=2, keepdims=True)
            img_normal = img_normal / (img_tmp + (img_tmp == 0).astype(np.float32))
            tmp = np.where(img_tmp > 0)
            u_min = max(np.min(tmp[1]) - 10, 0)
            u_max = min(np.max(tmp[1]) + 10, W)
            v_min = max(np.min(tmp[0]) - 10, 0)
            v_max = min(np.max(tmp[0]) + 10, H)
            img_crop = img_normal[v_min: v_max, u_min: u_max, :]

            # print(u_max - u_min, v_max - v_min)
            k = 1
            img_smooth = img_crop.copy()
            for u in range(u_max - u_min):
                for v in range(v_max - v_min):
                    if img_crop[v, u, 0] == 0:
                        u_idx_min = max(0, u - k)
                        u_idx_max = min(u + k + 1, u_max - u_min)
                        v_idx_min = max(0, v - k)
                        v_idx_max = min(v + k + 1, v_max - v_min)
                        normal = np.sum(np.reshape(img_crop[v_idx_min:v_idx_max, u_idx_min:u_idx_max, :], [-1, 3]),
                                        axis=0)
                        if np.linalg.norm(normal) > 0:
                            img_smooth[v, u] = normal / np.linalg.norm(normal)

            # plt.figure()
            # plt.imshow((img_smooth + 1) / 2)
            # plt.axis('off')
            # plt.show()
            # img_c = (cv2.cvtColor(cv2.imread(c_files[i]), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
            # plt.figure()
            # plt.imshow(img_c[v_min:v_max, u_min:u_max, :])
            # plt.axis('off')
            # plt.show()

            img_angle = np.zeros([v_max - v_min, u_max - u_min, 2])  # theta, phi
            img_angle[:, :, 0] = np.arccos(img_smooth[:, :, 2]) * (img_smooth[:, :, 2] != 0).astype(np.float32)
            img_angle[:, :, 1] = np.arctan2(img_smooth[:, :, 0], img_smooth[:, :, 1])
            img_angle = (img_angle * 1e4).astype(np.int16)

            img_angle = img_angle.astype(np.float32) / 1e4
            img_recover = np.zeros([v_max - v_min, u_max - u_min, 3])
            img_recover[:, :, 2] = np.cos(img_angle[:, :, 0]) * (img_angle[:, :, 0] != 0).astype(np.float32)
            img_recover[:, :, 0] = np.sin(img_angle[:, :, 0]) * np.sin(img_angle[:, :, 1])
            img_recover[:, :, 1] = np.sin(img_angle[:, :, 0]) * np.cos(img_angle[:, :, 1])
            # print(np.max(np.abs(img_smooth-img_recover)))

            # save files
            with open('%s/normal_%i.pkl' % (save_dir, i), 'wb') as f:
                pickle.dump([img_angle, v_min, v_max, u_min, u_max], f)

            # time4 = time.time()
            # print(time2-time1, time3-time2, time4-time3)
            #
            # plt.figure()
            # plt.imshow((img_recover + 1) / 2)
            # plt.axis('off')
            # plt.show()


'''

from scipy.spatial import Delaunay
from fitting.render_model import render_model


class Camera:
    def __init__(self, t, rt, f, c):
        self.t = t
        self.rt = rt
        self.f = f
        self.c = c

root_dir = '/home/data/data_shihao'
H, W = 1024, 1224

# load extrinsic params
extrinsic_subset = 'sub0906'
with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
    params0906 = pickle.load(f)
extrinsic_subset = 'sub0909'
with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
    params0909 = pickle.load(f)

filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))
for idx, filename in enumerate(filenames):
    if 'zoushihao_demo' not in filename:
        continue

    # load correpsonding camera params
    name = filename.split('_')[0]
    if name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting', 'guochuan']:
        param_p = params0906['param_p']
        param_d1 = params0906['param_d1']
        param_d2 = params0906['param_d2']
        param_d3 = params0906['param_d3']
        T_d1p = utils.convert_param2tranform(params0906['d1p'])
        T_pd1 = T_d1p.inv()
        T_d2p = utils.convert_param2tranform(params0906['d2p'])
        T_pd2 = T_d2p.inv()
        T_d3p = utils.convert_param2tranform(params0906['d3p'])
        T_pd3 = T_d3p.inv()
    else:
        param_p = params0909['param_p']
        param_d1 = params0909['param_d1']
        param_d2 = params0909['param_d2']
        param_d3 = params0909['param_d3']
        T_d1p = utils.convert_param2tranform(params0909['d1p'])
        T_pd1 = T_d1p.inv()
        T_d2p = utils.convert_param2tranform(params0909['d2p'])
        T_pd2 = T_d2p.inv()
        T_d3p = utils.convert_param2tranform(params0909['d3p'])
        T_pd3 = T_d3p.inv()

    flength = np.asarray(param_p[0:2])
    center = np.asarray(param_p[2:4])
    N = len(glob.glob('%s/annotation_openpose_kinect/fusion/%s/*.pkl' % (root_dir, filename)))
    print('working on %s (%i), %i examples' % (filename, idx, N))
    d1_files = ['%s/depth/PC1/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
    d2_files = ['%s/depth/PC2/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
    d3_files = ['%s/depth/PC3/%s/depth_%i.png' % (root_dir, filename, i) for i in range(N)]
    mesh_files = ['%s/fusion_depth_mesh/%s/depth_mesh_%i.ply' % (root_dir, filename, i) for i in range(N)]
    seg1_files = ['%s/segmentation_plane_fitting/PC1/%s/seg_depth_%i.mat' % (root_dir, filename, i) for i in range(N)]
    seg2_files = ['%s/segmentation_plane_fitting/PC2/%s/seg_depth_%i.mat' % (root_dir, filename, i) for i in range(N)]
    seg3_files = ['%s/segmentation_plane_fitting/PC3/%s/seg_depth_%i.mat' % (root_dir, filename, i) for i in range(N)]

    for i in range(N):
        if i != 400:
            continue
        print(mesh_files[i])
        img_file = '%s/PC4/%s/polar0-45_%i.jpg' % (root_dir, filename, i)
        img = cv2.imread(img_file)
        img[:, :, 2] = img[:, :, 1]
        h, w = 1024, 1224

        img_d1 = cv2.imread(d1_files[i], -1)[:, ::-1]
        img_d2 = cv2.imread(d2_files[i], -1)[:, ::-1]
        img_d3 = cv2.imread(d3_files[i], -1)[:, ::-1]

        seg_uv1 = loadmat(seg1_files[i])['uv'] - 1
        seg_uv2 = loadmat(seg2_files[i])['uv'] - 1
        seg_uv3 = loadmat(seg3_files[i])['uv'] - 1

        uvd1 = np.stack([seg_uv1[:, 0], seg_uv1[:, 1], img_d1[seg_uv1[:, 1], seg_uv1[:, 0]]], axis=1)
        uvd2 = np.stack([seg_uv2[:, 0], seg_uv2[:, 1], img_d2[seg_uv2[:, 1], seg_uv2[:, 0]]], axis=1)
        uvd3 = np.stack([seg_uv3[:, 0], seg_uv3[:, 1], img_d3[seg_uv3[:, 1], seg_uv3[:, 0]]], axis=1)

        img1 = np.zeros_like(img_d1)
        img1[uvd1[:, 1], uvd1[:, 0]] = uvd1[:, 2]
        img2 = np.zeros_like(img_d2)
        img2[uvd2[:, 1], uvd2[:, 0]] = uvd2[:, 2]
        img3 = np.zeros_like(img_d3)
        img3[uvd3[:, 1], uvd3[:, 0]] = uvd3[:, 2]

        uvd1 = connectComponent(img1, uvd1)
        uvd2 = connectComponent(img2, uvd2)
        uvd3 = connectComponent(img3, uvd3)

        xyz1 = utils.uvd2xyz(uvd1, param_d1)
        xyz2 = utils.uvd2xyz(uvd2, param_d2)
        xyz3 = utils.uvd2xyz(uvd3, param_d3)

        smpl_file = '%s/fitting_results/%s/smpl_sfd_%i.pkl' % (root_dir, filename, i)
        if os.path.exists(smpl_file):
            with open(smpl_file, 'rb') as f:
                param = pickle.load(f)

        for idx, xyz in enumerate([xyz1, xyz2, xyz3]):

            tri = Delaunay(xyz[:, 0:2])
            triangle_points = xyz[tri.simplices]  # [N, 3, 3 (x, y, z)]
            # remove long face
            triangle_dis = [np.sum((triangle_points[:, ii[0], :] - triangle_points[:, ii[1], :])**2, axis=1)
                            for ii in [[0, 1], [1, 2], [0, 2]]]
            tri_idx = (triangle_dis[0] < 50**2) & (triangle_dis[1] < 50**2) & (triangle_dis[2] < 50**2)

            if idx == 0:
                xyz_p = T_pd1.transform(utils.uvd2xyz(uvd1, param_d1)) / 1000
            elif idx == 1:
                xyz_p = T_pd2.transform(utils.uvd2xyz(uvd2, param_d2)) / 1000
            else:
                xyz_p = T_pd3.transform(utils.uvd2xyz(uvd3, param_d3)) / 1000

            vertex = np.array([(i[0], i[1], -i[2]) for i in xyz_p], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            face = np.array([(tuple(i), 255, 255, 255) for i in tri.simplices[tri_idx, :]],
                            dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            el = PlyElement.describe(vertex, 'vertex')
            el2 = PlyElement.describe(face, 'face')
            plydata = PlyData([el, el2])
            plydata.write('demo_%i.ply' % idx)

            dist = np.abs(np.mean(xyz_p, axis=0)[2])
            cam = Camera(np.array([0, 0, 0]), np.array([0, 0, 0]), param['cam_f'], param['cam_c'])
            im = (render_model(xyz_p, tri.simplices[tri_idx, :], w, h, cam, far=20 + dist, img=img) * 255.).astype('uint8')
            plt.figure(figsize=(12, 12))
            plt.imshow(im)
            plt.axis('off')
            plt.show()

            # faces = tri.simplices[tri_idx, :]
            # color = np.sum(xyz1[faces][:, :, 2], axis=1) / 3
            # plt.figure(figsize=(16, 16))
            # plt.tripcolor(xyz[:, 0], -xyz[:, 1], faces, facecolors=color, edgecolors='k')
            # plt.axis('off')
            # plt.colorbar()
            #
            # plt.savefig('Delaunay%i.pdf' % idx)
            # plt.show()


'''


def main(root_dir='/home/data/data_shihao', color_cam_num=2, num_cpus=18, computer=1):
    # load extrinsic params
    extrinsic_subset = 'sub0906'
    with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
        params0906 = pickle.load(f)
    extrinsic_subset = 'sub0909'
    with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
        params0909 = pickle.load(f)

    # filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))
    # obtain_normal_single_processor(filenames, root_dir, params0906, params0909, 0, color_cam_num)

    # filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))
    if computer == 1:  # shihao
        filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))[0:30]
    elif computer == 2:  # licheng1
        filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))[30:120]
    elif computer == 3:  # licheng3
        filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))[120:156]
    else:
        raise ValueError('computer errors...')
    N = len(filenames)
    n_files_cpu = N // num_cpus

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for i in range(num_cpus):
        idx1 = i * n_files_cpu
        idx2 = min((i + 1) * n_files_cpu, N)
        results.append(pool.apply_async(obtain_normal_single_processor,
                                        (filenames[idx1: idx2], root_dir, params0906, params0909, i, color_cam_num)))
    pool.close()
    pool.join()
    pool.terminate()

    for result in results:
        tmp = result.get()
        if tmp is not None:
            print(tmp)
    print('Multi-cpu pre-processing ends.')


def resize_normal(root_dir='/home/data/data_shihao', save_dir='/home/data2', img_size=256,
                  save_dir_mask='/data_shihao'):
    h, w = 1024, 1224

    filenames = sorted(os.listdir('%s/real_noisy_normal/' % root_dir))
    for filename in filenames:
        if not os.path.exists('%s/real_noisy_normal/%s' % (save_dir, filename)):
            os.mkdir('%s/real_noisy_normal/%s' % (save_dir, filename))
        print(filename)
        fitting_files = sorted(glob.glob('%s/real_noisy_normal/%s/normal_*.pkl' % (root_dir, filename)))
        for fname in fitting_files:
            # fname = '/home/data2/real_noisy_normal/zoushihao_demo/normal_100.pkl'
            # recover the normal of original size
            with open(fname, 'rb') as f:
                img_angle_crop, _v_min, _v_max, _u_min, _u_max = pickle.load(f)
                img_angle = np.zeros([h, w, 2])
                # angles are stored as "int16" by multiplying 1e4 to save disk space
                img_angle[_v_min: _v_max, _u_min: _u_max] = img_angle_crop.astype(np.float32) / 1e4

                xyz_normal = np.zeros([h, w, 3])
                xyz_normal[:, :, 2] = np.cos(img_angle[:, :, 0]) * (img_angle[:, :, 0] != 0).astype(np.float32)
                xyz_normal[:, :, 0] = np.sin(img_angle[:, :, 0]) * np.sin(img_angle[:, :, 1])
                xyz_normal[:, :, 1] = np.sin(img_angle[:, :, 0]) * np.cos(img_angle[:, :, 1])

            # read the mask file to get bbx
            fname_mask = fname.replace(root_dir, save_dir_mask).replace('real_noisy_normal', 'mask')\
                .replace('normal_', 'mask_')
            if os.path.exists(fname_mask):
                # mask file does not exist
                with open(fname_mask, 'rb') as f:
                    img_mask, (u_min, u_max, v_min, v_max, length) = pickle.load(f)

                xyz_normal = cv2.resize(xyz_normal[v_min:v_max, u_min:u_max, :], (img_size, img_size))
                norm = np.linalg.norm(xyz_normal, axis=2, keepdims=True)
                xyz_normal = xyz_normal / (norm + (norm == 0))

                # save disk space
                img_angle_resize = np.zeros([img_size, img_size, 2])  # theta, phi
                img_angle_resize[:, :, 0] = np.arccos(xyz_normal[:, :, 2]) * (norm[:, :, 0] > 0)
                img_angle_resize[:, :, 1] = np.arctan2(xyz_normal[:, :, 0], xyz_normal[:, :, 1])
                img_angle_resize = (img_angle_resize * 1e4).astype(np.int16)

                save_name = fname.replace(root_dir, save_dir)
                with open(save_name, 'wb') as f:
                    pickle.dump([img_angle_resize, (u_min, u_max, v_min, v_max, length)], f)

        #     import matplotlib.pyplot as plt
        #     plt.figure(figsize=(10, 10))
        #     plt.subplot(221)
        #     plt.imshow((xyz_normal + 1) / 2)
        #     plt.subplot(222)
        #     plt.imshow(np.expand_dims(img_mask, axis=2) * (xyz_normal + 1) / 2)
        #     plt.subplot(223)
        #     plt.imshow(img_mask, cmap='gray')
        #     plt.show()
        #     break
        # break


def move_depth():
    root_dir = '/home/data/data_shihao'
    filenames = sorted(os.listdir('%s/annotation_openpose_kinect/fusion' % root_dir))[120:156]
    for filename in filenames:
        cmd1 = 'cp -r %s/segmentation_plane_fitting/PC1/%s %s/to_shihao/segmentation_plane_fitting/PC1/' %\
               (root_dir, filename, root_dir)
        os.system(cmd1)
        cmd2 = 'cp -r %s/segmentation_plane_fitting/PC2/%s %s/to_shihao/segmentation_plane_fitting/PC2/' % \
               (root_dir, filename, root_dir)
        os.system(cmd2)
        cmd3 = 'cp -r %s/segmentation_plane_fitting/PC3/%s %s/to_shihao/segmentation_plane_fitting/PC3/' % \
               (root_dir, filename, root_dir)
        os.system(cmd3)

        cmd4 = 'cp -r %s/depth/PC1/%s %s/to_shihao/depth/PC1/' % (root_dir, filename, root_dir)
        os.system(cmd4)
        cmd5 = 'cp -r %s/depth/PC2/%s %s/to_shihao/depth/PC2/' % (root_dir, filename, root_dir)
        os.system(cmd5)
        cmd6 = 'cp -r %s/depth/PC3/%s %s/to_shihao/depth/PC3/' % (root_dir, filename, root_dir)
        os.system(cmd6)

        print('%s finish.' % filename)


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # move_depth()
    # main(root_dir='C:/to_shihao', num_cpus=6, color_cam_num=2, computer=1)
    main(root_dir='/home/data/data_shihao', num_cpus=10, color_cam_num=2, computer=2)
    # main(root_dir='/home/shihao/data', num_cpus=6, color_cam_num=2, computer=3)
    # resize_normal()

