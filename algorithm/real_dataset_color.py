import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import torch
import data_preprocessing.utils as utils


class PyREALDatasetColor(Dataset):
    def __init__(self, root_dir='/home/data/data_shihao', computer=1, cam_params_dir='../calib_new/calib_multi_data',
                 mode='train', task='all', normal_mode='color', shape_mode='1_normal_color',
                 img_size=256, pose_bbx=(256, 256, 0.8), use_smpl_mean=False):
        self.root_dir = root_dir
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        assert task in ['polar2normal', 'polar2shape', 'all']
        # for ablation study
        assert normal_mode in ['color']
        assert shape_mode in ['1_normal_color', '0_normal_color']
        self.task = task
        self.normal_mode = normal_mode
        self.shape_mode = shape_mode
        self.img_size = img_size
        self.computer = computer
        self.pose_bbx = np.asarray([pose_bbx])  # normalize uvd pose
        self.h, self.w = 1080, 1920

        # load extrinsic params
        self.cam_params = self.get_cam_param(cam_params_dir)

        # corresponding cam params to each subject
        self.name_cam_params = {}  # {"name": 0 or 1}
        for name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting', 'guochuan']:
            self.name_cam_params[name] = 0
        for name in ['duanyuwei', 'chenxiangye', 'cuihaibo', 'zhouming', 'ranxiaomin', 'yangji']:
            self.name_cam_params[name] = 1

        # corresponding cam params to each subject
        self.name_gender = {}  # {"name": 0 or 1}
        for name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting',
                     'guochuan',  'cuihaibo', 'yangji', 'duanyuwei']:
            self.name_gender[name] = 0
        for name in ['chenxiangye','zhouming', 'ranxiaomin']:
            self.name_gender[name] = 1

        # split train and test sets
        self.train_name_list = ['lyuxingzheng', 'yanhe', 'yangji', 'houpengyue', 'duanyuwei',
                                'chenxiangye', 'cuihaibo', 'zhouming']
        self.val_name_list = ['guochuan']
        self.test_name_list = ['ranxiaomin', 'zouting', 'zoushihao']

        if os.path.exists('%s/realdata_color_%s_files.pkl' % (self.root_dir, self.mode)):
            with open('%s/realdata_color_%s_files.pkl' % (self.root_dir, self.mode), 'rb') as f:
                self.all_color_files, self.all_poses = pickle.load(f)
        else:
            self.all_color_files, self.all_poses = self.obtain_all_filenames()

        self.smpl_mean = np.load('%s/smpl_mean.npy' % self.root_dir)
        if not use_smpl_mean:
            self.smpl_mean[:] = 0.

    def __len__(self):
        return len(self.all_color_files)

    def __getitem__(self, idx):
        fname = self.all_color_files[idx]  # '%s/color/PC2/%s/color_*.jpg' % (self.root_dir, filename)

        if self.mode == 'val':
            ppsd_fname = fname.replace('color/PC2', 'val_color_ppsd').\
                replace('/color_', '/data_').replace('.jpg', '.pkl')
            if self.computer == 1:
                ppsd_fname = ppsd_fname.replace(self.root_dir, '/data_shihao')
            with open(ppsd_fname, 'rb') as f:
                # # one_sample: {img, normal, mask, joint_uvd, joint_xyz, smpl_param, info}
                one_sample = pickle.load(f)

            if self.task == 'polar2shape':
                one_sample['normal'] = np.array([0])
                one_sample['mask'] = np.array([0])

            one_sample['smpl_param'] = one_sample['smpl_param'] - self.smpl_mean
            for key, item in one_sample.items():
                one_sample[key] = torch.from_numpy(item).float()
            return one_sample

        # load original mask and bbx
        fname_mask = fname.replace('color/PC2', 'color2_mask').replace('color_', 'mask_').replace('.jpg', '.pkl')
        if self.computer == 1:
            fname_mask = fname_mask.replace(self.root_dir, '/home/data2')
        with open(fname_mask, 'rb') as f:
            # bbx: [u_min, u_max, v_min, v_max]
            mask_crop, _u_min, _u_max, _v_min, _v_max = pickle.load(f)
            mask = np.zeros([self.h, self.w])
            mask[_v_min: _v_max, _u_min: _u_max] = mask_crop

            # expand the boundary
            _u_min = max(_u_min - 30, 0)
            _u_max = min(_u_max + 30, self.w)
            _v_min = max(_v_min - 30, 0)
            _v_max = min(_v_max + 30, self.h)

            length = max(_u_max - _u_min, _v_max - _v_min)
            u_min = int((_u_min + _u_max - length) / 2)
            u_max = int((_u_min + _u_max + length) / 2)
            v_min = int((_v_min + _v_max - length) / 2)
            v_max = int((_v_min + _v_max + length) / 2)
            if u_min < 0:
                u_min = 0
                u_max = length
            if u_max > self.w:
                u_min = self.w - length
                u_max = self.w
            if v_min < 0:
                v_min = 0
                v_max = length
            if v_max > self.h:
                v_min = self.h - length
                v_max = self.h
            bbx = [u_min, u_max, v_min, v_max]

            # resize mask
            mask = cv2.resize(mask[v_min:v_max, u_min:u_max].astype(np.float32), (self.img_size, self.img_size))
            mask = (mask == 1)

        # resize color image
        if self.computer == 1:
            fname_img = fname.replace(self.root_dir, '/home/data2')
        else:
            fname_img = fname
        img = (cv2.cvtColor(cv2.imread(fname_img), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
        img = cv2.resize(img[v_min:v_max, u_min:u_max].astype(np.float32), (self.img_size, self.img_size)) / 255.

        # one_sample: {img, normal, mask, joint_uvd, joint_xyz, smpl_param, info}
        one_sample = {}
        one_sample['img'] = np.transpose(img, [2, 0, 1])
        if self.task == 'polar2normal' or self.task == 'all':
            normal = self.get_normal(fname, mask)
            one_sample['normal'] = np.transpose(normal, [2, 0, 1])
            one_sample['mask'] = np.expand_dims(mask, axis=0)
        else:
            one_sample['normal'] = np.array([0])
            one_sample['mask'] = np.array([0])

        joint_uvd, smpl_param, info, joint_xyz = self.get_pose(idx, bbx)
        one_sample['joint_uvd'] = joint_uvd
        one_sample['joint_xyz'] = joint_xyz
        one_sample['smpl_param'] = smpl_param - self.smpl_mean
        one_sample['info'] = info

        for key, item in one_sample.items():
            one_sample[key] = torch.from_numpy(item).float()
        return one_sample

    def get_normal(self, fname, mask):
        # load target normal map
        fname_normal = fname.replace('color/PC2', 'color2_noisy_normal')\
            .replace('color_', 'normal_').replace('.jpg', '.pkl')
        if self.computer == 1:
            fname_normal = fname_normal.replace(self.root_dir, '/data_shihao')
        with open(fname_normal, 'rb') as f:
            img_angle_resize, _ = pickle.load(f)
            # angles are stored as "int16" by multiplying 1e4 to save disk space
            img_angle_resize = img_angle_resize.astype(np.float32) / 1e4
            _theta, _phi = img_angle_resize[:, :, 0], img_angle_resize[:, :, 1]
            z = np.cos(_theta) * mask
            x = np.sin(_theta) * np.sin(_phi)
            y = np.sin(_theta) * np.cos(_phi)
            xyz_normal = np.stack([x, y, z], axis=2)

        #     img_angle_crop, _v_min, _v_max, _u_min, _u_max = pickle.load(f)
        #     img_angle = np.zeros([self.h, self.w, 2])
        #     # angles are stored as "int16" by multiplying 1e4 to save disk space
        #     img_angle[_v_min: _v_max, _u_min: _u_max] = img_angle_crop.astype(np.float32) / 1e4
        #
        #     _theta, _phi = img_angle[:, :, 0], img_angle[:, :, 1]
        #     z = np.cos(_theta) * (_theta != 0).astype(np.float32)
        #     x = np.sin(_theta) * np.sin(_phi)
        #     y = np.sin(_theta) * np.cos(_phi)
        #     xyz_normal = np.stack([x, y, z], axis=2)
        #
        # u_min, u_max, v_min, v_max = bbx
        # xyz_normal = cv2.resize(xyz_normal[v_min:v_max, u_min:u_max, :], (self.img_size, self.img_size))
        # norm = np.linalg.norm(xyz_normal, axis=2, keepdims=True)
        # xyz_normal = xyz_normal / (norm + (norm == 0))
        return xyz_normal

    def get_pose(self, idx, bbx):
        # 3D pose and smpl shape parameters
        xyz_joint, smpl_param = self.all_poses[idx]
        u_min, u_max, v_min, v_max = bbx
        length = u_max - u_min

        # cam parameters
        color_cam_param = self.cam_params[self.name_cam_params[
            self.all_color_files[idx].split('/')[-2].split('_')[0]]]
        fx, fy, cx, cy, _, _, _ = color_cam_param[2]
        ratio = length / self.img_size
        cx = (cx - u_min) / ratio
        fx = fx / ratio
        cy = (cy - v_min) / ratio
        fy = fy / ratio
        cam_param = (fx, fy, cx, cy)

        # gender
        gender = self.name_gender[self.all_color_files[idx].split('/')[-2].split('_')[0]]

        # normalize pose
        uvd = self.project(xyz_joint, cam_param)
        root_d = uvd[0, 2]
        uvd[:, 2] = uvd[:, 2] - root_d
        uvd_joint = uvd / self.pose_bbx
        uvd_joint[:, 0:2] = 2 * uvd_joint[:, 0:2] - 1
        info = np.asarray(cam_param + (root_d,) + (gender,))

        # root_joint = pose_3d[0]  # [num_joint, 3]
        # pose_3d = pose_3d - root_joint
        # pose = pose_3d / self.pose_bbx
        # info = np.asarray(cam_param + tuple(root_joint) + (gender,))
        return uvd_joint, smpl_param, info, xyz_joint

    @staticmethod
    def get_cam_param(cam_params_dir):
        cam_params = []
        with open('%s/sub0906/extrinsic_params.pkl' % cam_params_dir, 'rb') as f:
            params0906 = pickle.load(f)
            T_d2p = utils.convert_param2tranform(params0906['d2p'])
            T_cd2 = utils.convert_param2tranform(params0906['cd2'])
            T_c2p = T_cd2 * T_d2p
            cam_param1 = [T_c2p.R, T_c2p.t, params0906['param_c2']]
            cam_params.append(cam_param1)
        with open('%s/sub0909/extrinsic_params.pkl' % cam_params_dir, 'rb') as f:
            params0909 = pickle.load(f)
            T_d2p = utils.convert_param2tranform(params0909['d2p'])
            T_cd2 = utils.convert_param2tranform(params0909['cd2'])
            T_c2p = T_cd2 * T_d2p
            cam_param2 = [T_c2p.R, T_c2p.t, params0909['param_c2']]
            cam_params.append(cam_param2)
        return cam_params

    def obtain_all_filenames(self):
        if self.mode == 'train':
            name_list = self.train_name_list
        elif self.mode == 'val':
            name_list = self.val_name_list
        elif self.mode == 'test':
            name_list = self.test_name_list
        else:
            raise ValueError('Unkonwn mode %s' % self.mode)

        print('[Obtain all filenames for %s]' % self.mode)
        all_color_files, all_poses, smpl_mean = [], [], []
        filenames = sorted(os.listdir('%s/color/PC2/' % self.root_dir))
        for idx, filename in enumerate(filenames):
            print('processed %i / %i' % (idx, len(filenames)))
            name = filename.split('_')[0]
            if name in name_list:
                cam_param = self.cam_params[self.name_cam_params[name]]  # [R, T, intrinsic]
                R, T = cam_param[0], cam_param[1]
                # read smpl shape parameters as dict
                shape_params = {}
                with open('%s/pose/%s/shape_smpl.txt' % (self.root_dir, filename), 'r') as f:
                    for line in f.readlines():
                        tmp = line.split(' ')
                        smpl_param = np.asarray([float(i) for i in tmp[1:]])
                        trans_c = (np.dot(smpl_param[0:3] * 1000, R.T) + T) / 1000
                        smpl_param = np.concatenate([smpl_param[3:13], trans_c, smpl_param[13:85]], axis=0)
                        shape_params[tmp[0]] = smpl_param

                # read 3D joint pose
                with open('%s/pose/%s/pose.txt' % (self.root_dir, filename), 'r') as f:
                    for line in f.readlines():
                        tmp = line.split(' ')
                        pose_polar = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3])  # [24, 3]
                        pose_color = (np.dot(pose_polar * 1000, R.T) + T) / 1000

                        # check whether its label (mask and normal) exits
                        fname = '%s/color/PC2/%s/color_%s.jpg' % (self.root_dir, filename, tmp[0])
                        fname_normal = fname.replace('color/PC2', 'color2_noisy_normal').replace('color_', 'normal_')\
                            .replace('.jpg', '.pkl')
                        fname_mask = fname.replace('color/PC2', 'mask').replace('color_', 'mask_')\
                            .replace('.jpg', '.pkl')
                        if os.path.exists(fname_normal) and os.path.exists(fname_mask):
                            all_poses.append([pose_color, shape_params[tmp[0]]])
                            smpl_mean.append(shape_params[tmp[0]])
                            all_color_files.append(fname)

        if self.mode == 'train':
            smpl_mean = np.mean(np.asarray(smpl_mean), axis=0)
            np.save('%s/smpl_mean' % self.root_dir, smpl_mean)

        with open('%s/realdata_color_%s_files.pkl' % (self.root_dir, self.mode), 'wb') as f:
            pickle.dump((all_color_files, all_poses), f)
        print('[%s: %i examples]' % (self.mode, len(all_color_files)))
        return all_color_files, all_poses

    @staticmethod
    def project(points, params):
        # points: [N, 3]
        fx, fy, cx, cy = params
        U = cx + fx * points[:, 0] / points[:, 2]
        V = cy + fy * points[:, 1] / points[:, 2]
        D = points[:, 2]
        return np.stack((U, V, D), axis=-1)  # N x 2

    def visualization(self, idx):
        one_sample = self.__getitem__(idx)

        # [H, W, C]
        img = np.transpose(one_sample['img'], [1, 2, 0])
        normal = np.transpose(one_sample['normal'], [1, 2, 0])
        mask = np.transpose(one_sample['mask'], [1, 2, 0])

        info = one_sample['info'].numpy()
        cam_param = info[0:4]
        xyz_pose = one_sample['joint_xyz'].numpy()
        pose = self.project(xyz_pose, cam_param)

        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(222)
        plt.imshow((mask * normal + 1) / 2)
        plt.axis('off')

        plt.subplot(223)
        plt.imshow((normal + 1) / 2)
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(mask[:, :, 0].numpy().astype(np.float32), cmap='gray')
        plt.axis('off')

        plt.show()

        kinematic_tree = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8],
                          [6, 9], [7, 10], [8, 11], [9, 12], [9, 14], [9, 13], [12, 15],
                          [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
        plt.figure()
        plt.imshow(img)
        plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=15)
        for idx1, idx2 in kinematic_tree:
            plt.plot([pose[idx1, 0], pose[idx2, 0]], [pose[idx1, 1], pose[idx2, 1]], color='b', linewidth=1.5)
        plt.show()


def preprocess_val_data(root_dir, computer):
    dataset = PyREALDatasetColor(root_dir=root_dir, computer=computer, mode='val', task='all')
    for idx, fname in enumerate(dataset.all_color_files):
        if idx % 2000 == 0:
            print(idx)

        # load original mask and bbx
        fname_mask = fname.replace('color/PC2', 'color2_mask').replace('color_', 'mask_').replace('.jpg', '.pkl')
        if dataset.computer == 1:
            fname_mask = fname_mask.replace(dataset.root_dir, '/home/data2')
        with open(fname_mask, 'rb') as f:
            # bbx: [u_min, u_max, v_min, v_max]
            mask_crop, _u_min, _u_max, _v_min, _v_max = pickle.load(f)
            mask = np.zeros([dataset.h, dataset.w])
            mask[_v_min: _v_max, _u_min: _u_max] = mask_crop

            # expand the boundary
            _u_min = max(_u_min - 30, 0)
            _u_max = min(_u_max + 30, dataset.w)
            _v_min = max(_v_min - 30, 0)
            _v_max = min(_v_max + 30, dataset.h)

            length = max(_u_max - _u_min, _v_max - _v_min)
            u_min = int((_u_min + _u_max - length) / 2)
            u_max = int((_u_min + _u_max + length) / 2)
            v_min = int((_v_min + _v_max - length) / 2)
            v_max = int((_v_min + _v_max + length) / 2)
            if u_min < 0:
                u_min = 0
                u_max = length
            if u_max > dataset.w:
                u_min = dataset.w - length
                u_max = dataset.w
            if v_min < 0:
                v_min = 0
                v_max = length
            if v_max > dataset.h:
                v_min = dataset.h - length
                v_max = dataset.h
            bbx = [u_min, u_max, v_min, v_max]

            # resize mask
            mask = cv2.resize(mask[v_min:v_max, u_min:u_max].astype(np.float32), (dataset.img_size, dataset.img_size))
            mask = (mask == 1)

        # resize color image
        if dataset.computer == 1:
            fname_img = fname.replace(dataset.root_dir, '/home/data2')
        else:
            fname_img = fname
        img = (cv2.cvtColor(cv2.imread(fname_img), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
        img = cv2.resize(img[v_min:v_max, u_min:u_max].astype(np.float32), (dataset.img_size, dataset.img_size)) / 255.

        # one_sample: {img, normal, mask, joint_uvd, joint_xyz, smpl_param, info}
        one_sample = {}
        one_sample['img'] = np.transpose(img, [2, 0, 1])
        if dataset.task == 'polar2normal' or dataset.task == 'all':
            normal = dataset.get_normal(fname, mask)
            one_sample['normal'] = np.transpose(normal, [2, 0, 1])
            one_sample['mask'] = np.expand_dims(mask, axis=0)
        else:
            one_sample['normal'] = np.array([0])
            one_sample['mask'] = np.array([0])

        joint_uvd, smpl_param, info, joint_xyz = dataset.get_pose(idx, bbx)
        one_sample['joint_uvd'] = joint_uvd
        one_sample['joint_xyz'] = joint_xyz
        one_sample['smpl_param'] = smpl_param
        one_sample['info'] = info

        for key, item in one_sample.items():
            if np.isinf(item).any():
                print('[error] inf %s, %s' % (key, fname))
            if np.isnan(item).any():
                print('[error] nan %s, %s' % (key, fname))
            if key in ['category', 'mask']:
                one_sample[key] = item.astype(np.int8)
            else:
                one_sample[key] = item.astype(np.float32)

        # fname: '%s/color/PC2/%s/color_%s.jpg' % (root_dir, filename, num)
        # save_fname: '%s/val_color_ppsd/%s/data_%s.pkl'
        name = fname.split('/')[-2]
        filename = '%s/val_color_ppsd/%s' % (dataset.root_dir, name)
        if computer == 1:
            filename = filename.replace(dataset.root_dir, '/data_shihao')
        if not os.path.exists(filename):
            os.mkdir(filename)

        save_fname = fname.replace('color/PC2', 'val_color_ppsd').replace('/color_', '/data_').replace('.jpg', '.pkl')
        if computer == 1:
            save_fname = save_fname.replace(dataset.root_dir, '/data_shihao')
        with open(save_fname, 'wb') as f:
            pickle.dump(one_sample, f)

        # print(save_fname)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    root_dir = '/home/data/data_shihao'
    computer = 1
    # root_dir = '/home/shihao/data_shihao'
    # computer = 2
    # preprocess_val_data(root_dir, computer)

    dataset_train = PyREALDatasetColor(root_dir=root_dir, computer=computer, mode='train', task='all')
    i = np.random.randint(0, len(dataset_train), 1)[0]
    print(i, dataset_train.all_color_files[i])
    dataset_train.visualization(i)
    # one_sample = dataset_train[i]
    # for k, v in one_sample.items():
    #     print(k, v.shape)
    #
    dataset_train = PyREALDatasetColor(root_dir=root_dir, computer=computer, mode='val', task='all')
    i = np.random.randint(0, len(dataset_train), 1)[0]
    print(i, dataset_train.all_color_files[i])
    dataset_train.visualization(i)
    # one_sample = dataset_train[i]
    # for k, v in one_sample.items():
    #     print(k, v.shape)
    #
    # dataset_train = PyREALDatasetColor(root_dir=root_dir, computer=computer, mode='test', task='all')
    # i = np.random.randint(0, len(dataset_train), 1)[0]
    # print(i, dataset_train.all_color_files[i])
    # dataset_train.visualization(i)
    # one_sample = dataset_train[i]
    # for k, v in one_sample.items():
    #     print(k, v.shape)


