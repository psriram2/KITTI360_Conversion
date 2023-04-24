import os
import numpy as np
from tqdm import tqdm

dir_calib = 'calib_2'
dir_image = 'image_2'
dir_label = 'label_2'
dir_velodyne = '/hdd/datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data'
dir_output = 'kitti360'
os.makedirs(dir_output, exist_ok=True)

def rename():
    def get_newname(name):
        img_id = get_img_id(name, True)
        new_name = '{:0>6d}{}'.format(img_id, name[-4:])
        return new_name 
    
    for split in ['training', 'testing']:
        for type in ['calib', 'label_2', 'image_2', 'velodyne']:
            dir_data = os.path.join(dir_output, split, type)
            name_input = sorted([name for name in os.listdir(dir_data)])
            name_output = [get_newname(name) for name in name_input]
            for i in range(len(name_input)):
                command = 'mv {} {}'.format(os.path.join(dir_data, name_input[i]), os.path.join(dir_data, name_output[i]))
                print(command)
                os.system(command)
                # quit()

def get_img_id(name, int_id=False):
    index = name[-14:-4]
    if int_id:
        index = int(index)
    return index

def create_split():
    # generate split txt
    path_calib = sorted([os.path.join(dir_calib, name) for name in os.listdir(dir_calib)])
    img_ids = np.array([get_img_id(path, True) for path in path_calib])
    data_num = len(path_calib)
    
    # split files
    # train: 25%, validation: 25%, test: 50% -> follow the distribution of kitti dataset
    dir_imgsets = os.path.join(dir_output, 'ImageSets')
    os.makedirs(dir_imgsets, exist_ok=True)
    cut = int(data_num * 0.5)
    split_train_valid = img_ids[:cut]
    split_test = img_ids[cut:]
    sample = (np.arange(len(split_train_valid)) % 2) == 0
    split_train = split_train_valid[sample]
    split_valid = split_train_valid[~sample]

    with open(os.path.join(dir_imgsets, 'train.txt'), 'w') as file:
        for i in range(len(split_train)):
            line = '{:0>6d}\n'.format(split_train[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'val.txt'), 'w') as file:
        for i in range(len(split_valid)):
            line = '{:0>6d}\n'.format(split_valid[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'trainval.txt'), 'w') as file:
        for i in range(len(split_train_valid)):
            line = '{:0>6d}\n'.format(split_train_valid[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'test.txt'), 'w') as file:
        for i in range(len(split_test)):
            line = '{}\n'.format(split_test[i])
            file.write(line)

def copy_data_with_ids(dir_input, dir_output, ids):
    def contain(i, array):
        a = array - i
        zero_num = np.sum(a == 0)
        if zero_num > 0:
            return True
        else:
            return False
    
    os.makedirs(dir_output, exist_ok=True)
    paths_input = sorted([os.path.join(dir_input, name) for name in os.listdir(dir_input)])
    for path in tqdm(paths_input):
        path_id = get_img_id(path, int_id=True)
        if contain(path_id, ids):
            path_output = os.path.join(dir_output, '{:0>6d}{}'.format(path_id, path[-4:]))
            command = 'cp {} {}'.format(path, path_output)
            os.system(command)

def split_data():
    split_trainval_txt = os.path.join(dir_output, 'ImageSets', 'trainval.txt')
    split_test_txt  = os.path.join(dir_output, 'ImageSets', 'test.txt')

    split_trainval = np.genfromtxt(split_trainval_txt).astype(np.int32)
    split_test  = np.genfromtxt(split_test_txt).astype(np.int32)
    
    # calib
    copy_data_with_ids(dir_calib, os.path.join(dir_output, 'training', 'calib'), split_trainval)
    copy_data_with_ids(dir_calib, os.path.join(dir_output, 'testing', 'calib'), split_test)

    # label
    copy_data_with_ids(dir_label, os.path.join(dir_output, 'training', 'label_2'), split_trainval)
    copy_data_with_ids(dir_label, os.path.join(dir_output, 'testing', 'label_2'), split_test)
    
    # image
    copy_data_with_ids(dir_image, os.path.join(dir_output, 'training', 'image_2'), split_trainval)
    copy_data_with_ids(dir_image, os.path.join(dir_output, 'testing', 'image_2'), split_test)

    # velodyne
    copy_data_with_ids(dir_velodyne, os.path.join(dir_output, 'training', 'velodyne'), split_trainval)
    copy_data_with_ids(dir_velodyne, os.path.join(dir_output, 'testing', 'velodyne'), split_test)

if __name__ == '__main__':
    # create_split()
    split_data()