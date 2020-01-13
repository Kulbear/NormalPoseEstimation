import os
import pickle
import glob
import numpy as np

root_dir = '/home/data/data_shihao'
filenames = sorted(os.listdir('%s/color2_noisy_normal' % root_dir))

for filename in filenames:
    normal_files = glob.glob('%s/color2_noisy_normal/%s/normal_*.pkl' % (root_dir, filename))
    for normal_file in normal_files:
        with open(normal_file, 'rb') as f:
            img_angle, v_min, v_max, u_min, u_max = pickle.load(f)
        img_angle = (img_angle * 1e4).astype(np.int16)

        # with open(normal_file, 'wb') as f:
        #     pickle.dump([img_angle, v_min, v_max, u_min, u_max], f)
    break