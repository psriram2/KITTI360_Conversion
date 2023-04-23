import os
import numpy as np

from skimage import io
from tqdm import tqdm

# local imports
from label_helpers import Annotation3D
from labels import id2label
from kitti_helper import get_kitti_annotations
from project import CameraPerspective
from calib_helpers import loadPerspectiveIntrinsic, loadCalibrationRigid


ROOT_DIR = "./"
RAW_IMAGE_DATA = "./data_2d_raw"
INSTANCE_SEG_DATA = "./data_2d_semantics/train"
POSES_DATA = "./data_poses"
GT_CALIB_DATA = "./calibration"

KITTI_IMAGE_FOLDER = "./image_2"
KITTI_LABEL_FOLDER = "./label_2"
KITTI_CALIB_FOLDER = "./calib_2"

CAM_ID = 1
CATEGORIES = ['car', 'person', 'bicycle']
MAX_N = 1000


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
    instance_ids = set()

    nrows, ncols = instance_seg.shape
    for i in range(nrows):
        for j in range(ncols):
            instance_ids.add(instance_seg[i][j])

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
            
            # camera(obj3d, frame)
            # vertices = np.asarray(obj3d.vertices_proj).T
            # points.append(np.asarray(obj3d.vertices_proj).T)
            # depths.append(np.asarray(obj3d.vertices_depth))
            # for line in obj3d.lines:
            #     v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
            #     uv, d = camera.project_vertices(np.asarray(v), frame)
            #     mask = np.logical_and(np.logical_and(d>0, uv[0]>0), uv[1]>0)
            #     mask = np.logical_and(np.logical_and(mask, uv[0]<image.shape[1]), uv[1]<image.shape[0])
            #     plt.plot(uv[0][mask], uv[1][mask], 'r.', linewidth=0.1, markersize=1)

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
        os.makedirs(KITTI_IMAGE_FOLDER)

    if not os.path.exists(KITTI_LABEL_FOLDER):
        os.makedirs(KITTI_LABEL_FOLDER)

    if not os.path.exists(KITTI_CALIB_FOLDER):
        os.makedirs(KITTI_CALIB_FOLDER)

        
    filePersIntrinsic = os.path.join(GT_CALIB_DATA, 'perspective.txt')
    Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
    print('Loaded %s' % filePersIntrinsic)
    proj_matrix = np.array(Tr[f'P_rect_0{CAM_ID}'][:3, :])
    # print("proj_matrix: ", proj_matrix.shape)
    print("camera intrinsic: \n", proj_matrix)
    # 1/0

    R0_rect = np.array(Tr[f'R_rect_0{CAM_ID}'])
    print("rectification rotation: \n", R0_rect)

    fileCameraToVelo = os.path.join(GT_CALIB_DATA, 'calib_cam_to_velo.txt')
    Tr = loadCalibrationRigid(fileCameraToVelo)
    print('Loaded %s' % fileCameraToVelo)
    velo_to_cam = np.linalg.inv(np.array(Tr))
    print("velo_to_cam: \n", velo_to_cam)

    all_seqs = os.listdir(RAW_IMAGE_DATA)
    for i in tqdm(range(len(all_seqs))):
        seq = all_seqs[i]
        if seq == ".DS_Store": # check for garbage files
            continue


        camera = CameraPerspective(ROOT_DIR, seq, CAM_ID)

        label3DBboxPath = os.path.join(ROOT_DIR, 'data_3d_bboxes/train')
        annotation3D = Annotation3D(label3DBboxPath, seq, posesDir=POSES_DATA)
        instance_3d_dict = create_instance_3d_dict(annotation3D)


        cam = 'image_%02d' % CAM_ID + '/data_rect/'

        all_imgs = os.listdir(os.path.join(RAW_IMAGE_DATA, seq, cam))

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
            
            # print("processing image " + img_name + " / frame " + str(frame))

            label_path = os.path.join(KITTI_LABEL_FOLDER, seq + f"_CAM{CAM_ID}" + img_name + '.txt')
            if os.path.exists(label_path):
                # 1/0 # should never happen
                pass

            with open(label_path, "w") as label_file:
                for anno in annos_3d:
                    output = get_kitti_annotations(anno, camera, seq, frame)
                    label_file.write(output + '\n')
        

            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = proj_matrix  # camera transform --> MAKE SURE THAT IMAGES ARE UNDER 'image_2'
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = R0_rect  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = velo_to_cam[:3, :] # should not be used for monocular 3d either.
            kitti_transforms['Tr_imu_to_velo'] = np.zeros((3, 4)) # Dummy values.
            calib_path = os.path.join(KITTI_CALIB_FOLDER, seq + f"_CAM{CAM_ID}" + img_name + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))


            image_path = os.path.join(KITTI_IMAGE_FOLDER, seq + f"_CAM{CAM_ID}" + img_name + '.png')
            curr_img = io.imread(os.path.join(RAW_IMAGE_DATA, seq, cam, all_imgs[j]))
            io.imsave(image_path, curr_img)














