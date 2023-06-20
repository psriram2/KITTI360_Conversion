import os
import numpy as np
from tqdm import tqdm

dir_calib = 'calib_2'
dir_image = 'image_2'
dir_label = 'label_2'
# dir_velodyne = '/hdd/datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data'
dir_velodyne = '/projects/perception/personals/pranav/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data'
dir_output = './'
os.makedirs(dir_output, exist_ok=True)

# OFFSET = 12000
OFFSET = 0

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
    # path_calib = sorted([os.path.join(dir_calib, name) for name in os.listdir(dir_calib)])
    # img_ids = np.array([get_img_id(path, True) for path in path_calib])
    # data_num = len(path_calib)


    # PRANAV REMOVE
    path_imgs = sorted([os.path.join(dir_image, name) for name in os.listdir(dir_image)])
    img_ids = np.array([get_img_id(path, True) for path in path_imgs])
    data_num = len(path_imgs)

    print("data num: ", data_num)
    # print("img ids: ", img_ids)
    # 1/0


    
    # split files
    # train: 25%, validation: 25%, test: 50% -> follow the distribution of kitti dataset
    dir_imgsets = os.path.join(dir_output, 'ImageSets')
    os.makedirs(dir_imgsets, exist_ok=True)
    dir_imgids = os.path.join(dir_output, 'ImageID')
    os.makedirs(dir_imgids, exist_ok=True)

    cut = int(data_num * 0.5)
    # cut = int(data_num)

    ids_train_valid = img_ids[:cut]
    ids_test = img_ids[cut:]
    # ids_test = img_ids[:cut]


    sets_train_valid = np.arange(len(ids_train_valid))
    sets_test = np.arange(len(ids_test))

    # sample = (sets_train_valid % 2) == 0
    sample = (sets_train_valid - (len(sets_train_valid) // 2)) <= 0
    ids_train = ids_train_valid[sample]
    ids_valid = ids_train_valid[~sample]
    sets_train = sets_train_valid[sample]
    sets_valid = sets_train_valid[~sample]

    with open(os.path.join(dir_imgsets, 'train.txt'), 'w') as file:
        for i in range(len(sets_train)):
            line = '{:0>6d}\n'.format(sets_train[i]+OFFSET)
            file.write(line)
    
    with open(os.path.join(dir_imgids, 'train.txt'), 'w') as file:
        for i in range(len(ids_train)):
            line = '{:0>6d}\n'.format(ids_train[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'val.txt'), 'w') as file:
        for i in range(len(sets_valid)):
            line = '{:0>6d}\n'.format(sets_valid[i]+OFFSET)
            file.write(line)
    
    with open(os.path.join(dir_imgids, 'val.txt'), 'w') as file:
        for i in range(len(ids_valid)):
            line = '{:0>6d}\n'.format(ids_valid[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'trainval.txt'), 'w') as file:
        for i in range(len(sets_train_valid)):
            line = '{:0>6d}\n'.format(sets_train_valid[i]+OFFSET)
            file.write(line)

    with open(os.path.join(dir_imgids, 'trainval.txt'), 'w') as file:
        for i in range(len(ids_train_valid)):
            line = '{:0>6d}\n'.format(ids_train_valid[i])
            file.write(line)

    with open(os.path.join(dir_imgsets, 'test.txt'), 'w') as file:
        for i in range(len(sets_test)):
            line = '{:0>6d}\n'.format(sets_test[i]+OFFSET)
            file.write(line)
    
    with open(os.path.join(dir_imgids, 'test.txt'), 'w') as file:
        for i in range(len(ids_test)):
            line = '{:0>6d}\n'.format(ids_test[i])
            file.write(line)

def copy_data_with_ids(dir_input, dir_output, sets, ids):
    def match_set_with_id(i, sets, ids):
        match = (ids - i) == 0
        if np.sum(match) == 0:
            return -1
        else:
            return sets[match].item()
    
    os.makedirs(dir_output, exist_ok=True)
    paths_input = sorted([os.path.join(dir_input, name) for name in os.listdir(dir_input)])
    for path in tqdm(paths_input):
        path_id = get_img_id(path, int_id=True)

        # PRANAV REMOVE
        # path_id += OFFSET

        s = match_set_with_id(path_id, sets, ids)
        if s >= 0:
            path_output = os.path.join(dir_output, '{:0>6d}{}'.format(s, path[-4:]))
            command = 'cp {} {}'.format(path, path_output)
            os.system(command)

def split_data():
    sets_trainval_txt = os.path.join(dir_output, 'ImageSets', 'trainval.txt')
    sets_test_txt  = os.path.join(dir_output, 'ImageSets', 'test.txt')
    ids_trainval_txt = os.path.join(dir_output, 'ImageID', 'trainval.txt')
    ids_test_txt  = os.path.join(dir_output, 'ImageID', 'test.txt')

    sets_trainval = np.genfromtxt(sets_trainval_txt).astype(np.int32)
    sets_test  = np.genfromtxt(sets_test_txt).astype(np.int32)
    ids_trainval = np.genfromtxt(ids_trainval_txt).astype(np.int32)
    ids_test  = np.genfromtxt(ids_test_txt).astype(np.int32)
    
    # calib
    copy_data_with_ids(dir_calib, os.path.join(dir_output, 'training', 'calib'), sets_trainval, ids_trainval)
    copy_data_with_ids(dir_calib, os.path.join(dir_output, 'testing', 'calib'), sets_test, ids_test)

    # label
    copy_data_with_ids(dir_label, os.path.join(dir_output, 'training', 'label_2'), sets_trainval, ids_trainval)
    copy_data_with_ids(dir_label, os.path.join(dir_output, 'testing', 'label_2'), sets_test, ids_test)
    
    # image
    copy_data_with_ids(dir_image, os.path.join(dir_output, 'training', 'image_2'), sets_trainval, ids_trainval)
    copy_data_with_ids(dir_image, os.path.join(dir_output, 'testing', 'image_2'), sets_test, ids_test)

    # velodyne
    copy_data_with_ids(dir_velodyne, os.path.join(dir_output, 'training', 'velodyne'), sets_trainval, ids_trainval)
    copy_data_with_ids(dir_velodyne, os.path.join(dir_output, 'testing', 'velodyne'), sets_test, ids_test)

if __name__ == '__main__':
    create_split()
    split_data()
