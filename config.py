
config = {
    # "ROOT_DIR": "../KITTI360_scripts/", 
    "ROOT_DIR": "/projects/bbuq/psriram2/kitti360/KITTI-360/", 
    "RAW_IMAGE_DATA": "data_2d_raw",
    "INSTANCE_SEG_DATA": "data_2d_semantics/train",
    "POSES_DATA": "data_poses",
    "GT_CALIB_DATA": "calibration",
    "KITTI_IMAGE_FOLDER": "./image_2",
    "KITTI_LABEL_FOLDER": "./label_2",
    "KITTI_CALIB_FOLDER": "./calib_2",
    "KITTI_SUBSEQUENCE_FOLDER": "./subsample_2",
    "CAM_ID": 0,
    "SEQUENCE": [0,2,3,4,5,6,7,9,10],
    # "SEQUENCE": [0],
    "SUBSAMPLE_SIZE": 100,
    "NUM_SUBSEQUENCES": 3,
    "CATEGORIES": ['car', 'person', 'bicycle'],
    "MAX_N": 1000
}

# /projects/perception/datasets/KITTI-360