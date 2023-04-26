import os
import numpy as np

from skimage import io
from tqdm import tqdm

# local imports
from label_helpers import Annotation3D
from labels import id2label
from kitti_helper import get_kitti_annotations, estimate_ground_plane
from project import CameraPerspective
from calib_helpers import loadPerspectiveIntrinsic, loadCalibrationRigid
from config import config


ROOT_DIR = config["ROOT_DIR"]
RAW_IMAGE_DATA = config["RAW_IMAGE_DATA"]
INSTANCE_SEG_DATA = config["INSTANCE_SEG_DATA"]
POSES_DATA = config["POSES_DATA"]
GT_CALIB_DATA = config["GT_CALIB_DATA"]

KITTI_IMAGE_FOLDER = config["KITTI_IMAGE_FOLDER"]
KITTI_LABEL_FOLDER = config["KITTI_LABEL_FOLDER"]
KITTI_CALIB_FOLDER = config["KITTI_CALIB_FOLDER"]

CAM_ID = config["CAM_ID"]
SEQUENCE = config["SEQUENCE"]
CATEGORIES = config["CATEGORIES"]
MAX_N = config["MAX_N"]


def get_instance_map_path(sequence, frame):
    """creates the instance segmentation map path given the sequence and frame"""

    instance_folder = os.path.join(ROOT_DIR, INSTANCE_SEG_DATA, sequence, 'image_%02d' % CAM_ID, 'instance')
    instance_file = os.path.join(instance_folder, '%010d.png'%frame)
    return instance_file


def check_for_instance_segmentation(sequence, frame):
    """checks whether there is an instance segmentation map corresponding to given image"""

    instance_file = get_instance_map_path(sequence, frame)
    if os.path.exists(instance_file):
        return True
    
    return False


def get_instance_ids(sequence, frame):
    """gets the global instance ids in the current instance segmentation map"""

    instance_file = get_instance_map_path(sequence, frame)
    instance_seg = io.imread(instance_file)
    # instance_ids = set()
    # nrows, ncols = instance_seg.shape
    # for i in range(nrows):
    #     for j in range(ncols):
    #         instance_ids.add(instance_seg[i][j])
    instance_ids = set(instance_seg.flatten())

    return instance_ids


def create_instance_3d_dict(annotation3D):
    """Creates a dictionary to map global sinstance IDs (keys) to 3d bbox annotation objects (values)"""

    instance_3d_dict = {}

    for k, v in annotation3D.objects.items():
        if len(v.keys())==1 and (-1 in v.keys()): # only static objects
            obj3d = v[-1]
            if not id2label[obj3d.semanticId].name=='car': # only our desired categories
                continue

            global_id = obj3d.semanticId * MAX_N + obj3d.instanceId

            if global_id not in instance_3d_dict:
                instance_3d_dict[global_id] = obj3d

    return instance_3d_dict


def get_annos_3d(instance_3d_dict, instance_ids):
    """Given a list of instance ids, get the 3d annotations for each instance"""

    annos_3d = []
    for instance_id in instance_ids:
        if instance_id in instance_3d_dict:
            annos_3d.append(instance_3d_dict[instance_id])

    return annos_3d



if __name__ == "__main__":

    if not os.path.exists(KITTI_IMAGE_FOLDER):
        os.makedirs(KITTI_IMAGE_FOLDER, exist_ok=True)

    if not os.path.exists(KITTI_LABEL_FOLDER):
        os.makedirs(KITTI_LABEL_FOLDER, exist_ok=True)

    if not os.path.exists(KITTI_CALIB_FOLDER):
        os.makedirs(KITTI_CALIB_FOLDER, exist_ok=True)

        
    filePersIntrinsic = os.path.join(ROOT_DIR, GT_CALIB_DATA, 'perspective.txt')
    Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
    print('Loaded %s' % filePersIntrinsic)
    proj_matrix = np.array(Tr[f'P_rect_0{CAM_ID}'][:3, :])
    print("camera intrinsic: \n", proj_matrix)

    R0_rect = np.array(Tr[f'R_rect_0{CAM_ID}'])
    print("rectification rotation: \n", R0_rect)

    fileCameraToVelo = os.path.join(ROOT_DIR, GT_CALIB_DATA, 'calib_cam_to_velo.txt')
    Tr = loadCalibrationRigid(fileCameraToVelo)
    print('Loaded %s' % fileCameraToVelo)
    velo_to_cam = np.linalg.inv(np.array(Tr))
    print("velo_to_cam: \n", velo_to_cam)


    seq = '2013_05_28_drive_{:0>4d}_sync'.format(SEQUENCE)
    camera = CameraPerspective(ROOT_DIR, seq, CAM_ID)

    label3DBboxPath = os.path.join(ROOT_DIR, 'data_3d_bboxes/train')
    pose_dir = os.path.join(ROOT_DIR, POSES_DATA)
    annotation3D = Annotation3D(label3DBboxPath, seq, posesDir=pose_dir)
    instance_3d_dict = create_instance_3d_dict(annotation3D)

    cam = 'image_%02d' % CAM_ID + '/data_rect/'
    all_imgs = os.listdir(os.path.join(ROOT_DIR, RAW_IMAGE_DATA, seq, cam))

    for j in tqdm(range(len(all_imgs))):
        img = all_imgs[j]
        img_name = img[:-4]
        frame = int(img_name)

        if not check_for_instance_segmentation(seq, frame):
            continue

        instance_ids = get_instance_ids(seq, frame)
        annos_3d = get_annos_3d(instance_3d_dict, instance_ids)
        if len(annos_3d) == 0:  # make sure we have 3d annotations
            continue
           
        ground_plane = estimate_ground_plane(annos_3d, camera, frame)
        
        label_path = os.path.join(KITTI_LABEL_FOLDER, seq + f"_CAM{CAM_ID}_" + img_name + '.txt')
        if os.path.exists(label_path):
            # 1/0 # should never happen
            pass
        with open(label_path, "w") as label_file:
            for anno in annos_3d:
                output = get_kitti_annotations(anno, camera, seq, frame, ground_plane=ground_plane)
                if output != '':
                    label_file.write(output + '\n')
                
        if ground_plane is not None:
            proj_matrix[:3, :3] = np.matmul(proj_matrix[:3, :3], np.linalg.inv(ground_plane))
    
        kitti_transforms = dict()
        kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P2'] = proj_matrix  # camera transform --> MAKE SURE THAT IMAGES ARE UNDER 'image_2'
        kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['R0_rect'] = R0_rect  # Cameras are already rectified.
        kitti_transforms['Tr_velo_to_cam'] = velo_to_cam[:3, :] # should not be used for monocular 3d either.
        kitti_transforms['Tr_imu_to_velo'] = np.zeros((3, 4)) # Dummy values.
        calib_path = os.path.join(KITTI_CALIB_FOLDER, seq + f"_CAM{CAM_ID}_" + img_name + '.txt')
        with open(calib_path, "w") as calib_file:
            for (key, val) in kitti_transforms.items():
                val = val.flatten()
                val_str = '%.12e' % val[0]
                for v in val[1:]:
                    val_str += ' %.12e' % v
                calib_file.write('%s: %s\n' % (key, val_str))
        
        image_path = os.path.join(KITTI_IMAGE_FOLDER, seq + f"_CAM{CAM_ID}_" + img_name + '.png')
        curr_img = io.imread(os.path.join(ROOT_DIR, RAW_IMAGE_DATA, seq, cam, all_imgs[j]))
        io.imsave(image_path, curr_img)












