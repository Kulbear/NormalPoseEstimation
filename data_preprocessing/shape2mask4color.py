import sys
sys.path.append('../')
import cv2
import pickle
import os
from data_preprocessing import utils as utils
import glob
import numpy as np
import multiprocessing


def obtain_mask_single_processor(filenames, root_dir, params0906, params0909, cpu_i, color_cam_num=2):
    # load smpl model
    MODEL_DIR = '../calib_new/'
    MODEL_MALE_PATH = MODEL_DIR + "basicModel_m_lbs_10_207_0_v1.0.0.pkl"
    MODEL_FEMALE_PATH = MODEL_DIR + "basicModel_f_lbs_10_207_0_v1.0.0.pkl"

    H, W = 1080, 1920
    for idx, filename in enumerate(filenames):
        print('[cpu %2d, %i-th file] %s' % (cpu_i, idx, filename))
        # if 'guochuan_demo' not in filename:
        #     continue

        # load correpsonding camera params
        name = filename.split('_')[0]
        if name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting', 'guochuan']:
            if color_cam_num == 1:
                T_d1p = utils.convert_param2tranform(params0906['d1p'])
                T_cd1 = utils.convert_param2tranform(params0906['cd1'])
                T_cp = T_cd1 * T_d1p
                param_c = params0906['param_c1']
            elif color_cam_num == 2:
                T_d2p = utils.convert_param2tranform(params0906['d2p'])
                T_cd2 = utils.convert_param2tranform(params0906['cd2'])
                T_cp = T_cd2 * T_d2p
                param_c = params0906['param_c2']
            elif color_cam_num == 3:
                T_d3p = utils.convert_param2tranform(params0906['d3p'])
                T_cd3 = utils.convert_param2tranform(params0906['cd3'])
                T_cp = T_cd3 * T_d3p
                param_c = params0906['param_c3']
            else:
                raise ValueError('color camera num errors %i. It should be 1, 2 or 3.' % color_cam_num)
        else:
            if color_cam_num == 1:
                T_d1p = utils.convert_param2tranform(params0909['d1p'])
                T_cd1 = utils.convert_param2tranform(params0909['cd1'])
                T_cp = T_cd1 * T_d1p
                param_c = params0909['param_c1']
            elif color_cam_num == 2:
                T_d2p = utils.convert_param2tranform(params0909['d2p'])
                T_cd2 = utils.convert_param2tranform(params0909['cd2'])
                T_cp = T_cd2 * T_d2p
                param_c = params0909['param_c2']
            elif color_cam_num == 3:
                T_d3p = utils.convert_param2tranform(params0909['d3p'])
                T_cd3 = utils.convert_param2tranform(params0909['cd3'])
                T_cp = T_cd3 * T_d3p
                param_c = params0909['param_c3']
            else:
                raise ValueError('color camera num errors %i. It should be 1, 2 or 3.' % color_cam_num)

        if name in ['chenxiangye', 'zhouming', 'ranxiaomin']:
            model = utils.load_model(MODEL_FEMALE_PATH)
        else:
            model = utils.load_model(MODEL_MALE_PATH)

        save_dir = '%s/color%i_mask/%s' % (root_dir, color_cam_num, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fitting_files = sorted(glob.glob('%s/fitting_results/%s/smpl_sfd_*.pkl' % (root_dir, filename)))
        for fname in fitting_files:
            with open(fname, 'rb') as f:
                param = pickle.load(f)

            try:
                cam_t = T_cp.transform(param['cam_t'] * 1000) / 1000
                verts = T_cp.transform((param['verts'] + param['cam_t']) * 1000) / 1000 - cam_t
                dist = np.abs(cam_t[2] - np.mean(verts, axis=0)[2])
                cam_param = param_c[0:4]
                mask = (utils.render_model(verts, model.f, W, H, cam_param, cam_t, np.array([0, 0, 0]),
                                           far=20 + dist, img=None) * 255.).astype('uint8')

                mask = mask[:, :, 0] < 255
                tmp = np.where(mask == 1)
                u_min = max(np.min(tmp[1]) - 10, 0)
                u_max = min(np.max(tmp[1]) + 10, W)
                v_min = max(np.min(tmp[0]) - 10, 0)
                v_max = min(np.max(tmp[0]) + 10, H)
                mask = mask[v_min:v_max, u_min:u_max]

                # fname_img = fname.replace('fitting_results', 'color/PC%i' % color_cam_num).replace('smpl_sfd', 'color')\
                #     .replace('.pkl', '.jpg')
                # img_c = (cv2.cvtColor(cv2.imread(fname_img), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.float32) / 255.
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(img_c * np.expand_dims(mask.astype(np.float32), axis=2))
                # plt.show()

                # mask = cv2.resize(mask[:, 100:1124], (img_size, img_size))
                # mask = mask[:, :, 0] < 255

                save_fname = fname.replace('fitting_results', 'color%i_mask' % color_cam_num).replace('smpl_sfd', 'mask')
                # print(save_fname)
                with open(save_fname, 'wb') as f:
                    pickle.dump([mask, u_min, u_max, v_min, v_max], f)
            except:
                print('[error] %s' % fname)


def main(root_dir='/home/data/data_shihao', color_cam_num=2, num_cpus=18):
    # load extrinsic params
    extrinsic_subset = 'sub0906'
    with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
        params0906 = pickle.load(f)
    extrinsic_subset = 'sub0909'
    with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
        params0909 = pickle.load(f)

    filenames = sorted(os.listdir('%s/fitting_results/' % root_dir))
    # obtain_mask_single_processor(filenames, root_dir, params0906, params0909, 0, color_cam_num)
    N = len(filenames)
    n_files_cpu = N // num_cpus

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for i in range(num_cpus):
        idx1 = i * n_files_cpu
        idx2 = min((i + 1) * n_files_cpu, N)
        results.append(pool.apply_async(obtain_mask_single_processor,
                                        (filenames[idx1: idx2], root_dir, params0906, params0909, i, color_cam_num)))
    pool.close()
    pool.join()
    pool.terminate()

    for result in results:
        tmp = result.get()
        if tmp is not None:
            print(tmp)
    print('Multi-cpu pre-processing ends.')


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # move_depth()
    main(root_dir='/home/data/data_shihao', num_cpus=12, color_cam_num=2)
    # resize_normal()

