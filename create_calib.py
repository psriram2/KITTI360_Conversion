
import os
import numpy as np

from calib_helpers import loadPerspectiveIntrinsic, loadCalibrationRigid


raw_data = "./data_2d_raw"
gt_calib_data = "./calibration"

calib_folder = "./calib_2"
if not os.path.exists(calib_folder):
   os.makedirs(calib_folder)


filePersIntrinsic = os.path.join(gt_calib_data, 'perspective.txt')
Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
print('Loaded %s' % filePersIntrinsic)
proj_matrix = np.array(Tr['P_rect_00'][:3, :])
print("camera intrinsic: \n", proj_matrix)

R0_rect = np.array(Tr['R_rect_00'])
print("rectification rotation: \n", R0_rect)

fileCameraToVelo = os.path.join(gt_calib_data, 'calib_cam_to_velo.txt')
Tr = loadCalibrationRigid(fileCameraToVelo)
print('Loaded %s' % fileCameraToVelo)
velo_to_cam = np.linalg.inv(np.array(Tr))
print("velo_to_cam: \n", velo_to_cam)

1/0

# example path: data_2d_raw/2013_05_28_drive_{seq:0>4}_sync/image_{00|01}/data_rect/{frame:0>10}.png
for seq in os.listdir(raw_data):
    cam = "image_00/data_rect/"
    for img in os.listdir(os.path.join(raw_data, seq, cam)):
        img_name = img[:-4]
        print(f"processing image {img_name}.")

        kitti_transforms = dict()
        kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P2'] = proj_matrix  # camera transform --> MAKE SURE THAT IMAGES ARE UNDER 'image_2'
        kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['R0_rect'] = R0_rect  # Cameras are already rectified.
        kitti_transforms['Tr_velo_to_cam'] = velo_to_cam # should not be used for monocular 3d either.
        kitti_transforms['Tr_imu_to_velo'] = np.zeros((3, 4)) # Dummy values.
        calib_path = os.path.join(calib_folder, img_name + '.txt')
        with open(calib_path, "w") as calib_file:
            for (key, val) in kitti_transforms.items():
                val = val.flatten()
                val_str = '%.12e' % val[0]
                for v in val[1:]:
                    val_str += ' %.12e' % v
                calib_file.write('%s: %s\n' % (key, val_str))



# velodyne scans: data_3d_raw/2013_05_28_drive_{seq:0>4}_sync/velodyne_points/data/{frame:0>10}.bin
