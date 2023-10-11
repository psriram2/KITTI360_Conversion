import os
import numpy as np

from skimage import io

# local imports
from label_helpers import Annotation3D
from labels import id2label
from config import config
from matplotlib import pyplot as plt


ROOT_DIR = config["ROOT_DIR"]
RAW_IMAGE_DATA = config["RAW_IMAGE_DATA"]
INSTANCE_SEG_DATA =  config["INSTANCE_SEG_DATA"]
POSES_DATA = config["POSES_DATA"]

KITTI_LABEL_FOLDER = config["KITTI_LABEL_FOLDER"]

CAM_ID = config["CAM_ID"]
KITTI_CATEGORIES = {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}
MAX_N = config["MAX_N"]

IMAGE_H = 376
IMAGE_W = 1408
MAX_BOX_DIST = 40

def get_instance_map_path(sequence, frame):
    """creates the instance segmentation map path given the sequence and frame"""

    instance_folder = os.path.join(ROOT_DIR, INSTANCE_SEG_DATA, sequence, 'image_%02d' % CAM_ID, 'instance')
    instance_file = os.path.join(instance_folder, '%010d.png'%frame)
    return instance_file


def get_camera_pose(camera, frameId):
    if frameId not in camera.cam2world:
        last_k = None
        for k in sorted(camera.cam2world.keys()):
            if k > frameId:
                if last_k is not None:
                    return camera.cam2world[last_k]
                else:
                    1/0 # should never happen
                    return camera.cam2world[sorted(camera.cam2world.keys())[0]]
                
            last_k = k

    return camera.cam2world[frameId]


def remove_pitch(points_local):
    """Accepts the set of vertices of the 3d bbox and removes any pitch."""
    points_local_copy = np.copy(points_local)
    
    min_y = np.min(points_local[:, 1])
    max_y = np.max(points_local[:, 1])

    high_xz = np.array([points_local_copy[0], points_local_copy[2], points_local_copy[5], points_local_copy[7]])
    high_xz[:, 1] = min_y

    low_xz = np.array([points_local_copy[1], points_local_copy[3], points_local_copy[4], points_local_copy[6]])
    low_xz[:, 1] = max_y
    
    x1 = high_xz[0] - high_xz[2]
    x2 = high_xz[1] - high_xz[3]
    x3 = low_xz[0] - low_xz[2]
    x4 = low_xz[1] - low_xz[3]

    unit_l_dir = (x1 + x2 + x3 + x4) / np.linalg.norm(x1+x2+x3+x4)
    unit_h_dir = np.array([0, 1, 0])
    unit_w_dir = np.cross(unit_l_dir, unit_h_dir)

    proj_points = np.array([high_xz[0], low_xz[0], high_xz[1], low_xz[1], low_xz[2], high_xz[2], low_xz[3], high_xz[3]])

    # project all 8 points along length
    l_points = np.array([np.dot(a, unit_l_dir) for a in proj_points])
    l_min = np.argmin(l_points)
    l_max = np.argmax(l_points)


    proj_points[4] = proj_points[4] + (l_points[l_min]*unit_l_dir - l_points[4]*unit_l_dir)
    proj_points[5] = proj_points[5] + (l_points[l_min]*unit_l_dir - l_points[5]*unit_l_dir)
    proj_points[6] = proj_points[6] + (l_points[l_min]*unit_l_dir - l_points[6]*unit_l_dir)
    proj_points[7] = proj_points[7] + (l_points[l_min]*unit_l_dir - l_points[7]*unit_l_dir)

    proj_points[0] = proj_points[0] + (l_points[l_max]*unit_l_dir - l_points[0]*unit_l_dir)
    proj_points[1] = proj_points[1] + (l_points[l_max]*unit_l_dir - l_points[1]*unit_l_dir)
    proj_points[2] = proj_points[2] + (l_points[l_max]*unit_l_dir - l_points[2]*unit_l_dir)
    proj_points[3] = proj_points[3] + (l_points[l_max]*unit_l_dir - l_points[3]*unit_l_dir)
    
    # project all 8 points along width
    w_points = np.array([np.dot(a, unit_w_dir) for a in proj_points])
    w_min = np.argmin(w_points)
    w_max = np.argmax(w_points)
    
    proj_points[2] = proj_points[2] + (w_points[w_min]*unit_w_dir - w_points[2]*unit_w_dir)
    proj_points[3] = proj_points[3] + (w_points[w_min]*unit_w_dir - w_points[3]*unit_w_dir)
    proj_points[6] = proj_points[6] + (w_points[w_min]*unit_w_dir - w_points[6]*unit_w_dir)
    proj_points[7] = proj_points[7] + (w_points[w_min]*unit_w_dir - w_points[7]*unit_w_dir)

    proj_points[0] = proj_points[0] + (w_points[w_max]*unit_w_dir - w_points[0]*unit_w_dir)
    proj_points[1] = proj_points[1] + (w_points[w_max]*unit_w_dir - w_points[1]*unit_w_dir)
    proj_points[4] = proj_points[4] + (w_points[w_max]*unit_w_dir - w_points[4]*unit_w_dir)
    proj_points[5] = proj_points[5] + (w_points[w_max]*unit_w_dir - w_points[5]*unit_w_dir)

    return proj_points


def corrected_bbox(points_local):
    """
    Get new vertices of bbox (8, 3) in camera coordinate (OpenCV), 
    which is axis-aligned
            5 -------- 7
           /|         /|
          0 -------- 2 .
          | |        | |
          . 4 -------- 6
          |/         |/
          1 -------- 3
    """
    x = points_local[[0, 1, 2, 3]] - points_local[[5, 4, 7, 6]] #(4, 3)
    v_x = np.mean(x, axis=0)
    v_y = np.array([0, 1, 0])
    v_z = np.cross(v_x, v_y)
    v_z = v_z / np.linalg.norm(v_z)
    v_x = np.cross(v_y, v_z)
    v_x = v_x / np.linalg.norm(v_x)
    rot = np.stack([v_x, v_y, v_z])
    points_transformed = points_local @ rot.T

    points_new = np.zeros_like(points_local)
    points_new[[5, 4, 7, 6], 0] = np.min(points_transformed[:, 0])
    points_new[[0, 1, 2, 3], 0] = np.max(points_transformed[:, 0])
    points_new[[0, 2, 5, 7], 1] = np.min(points_transformed[:, 1])
    points_new[[1, 3, 4, 6], 1] = np.max(points_transformed[:, 1])
    points_new[[0, 1, 4, 5], 2] = np.min(points_transformed[:, 2])
    points_new[[2, 3, 6, 7], 2] = np.max(points_transformed[:, 2])

    points_new = points_new @ rot
    return points_new

def get_bbox_atttribute(points_bbox):
    """Get (h, w, l, x, y, z, rot_y, alpha) from bbox vertices"""
    h = np.linalg.norm(points_bbox[0] - points_bbox[1])
    l = np.linalg.norm(points_bbox[0] - points_bbox[5])
    w = np.linalg.norm(points_bbox[0] - points_bbox[2])
    x, y, z = np.mean(points_bbox[[1, 3, 4, 6]], axis=0)
    v_x = points_bbox[0] - points_bbox[5]
    rot_y = -np.arctan(v_x[2]/v_x[0])
    rot_view = np.arctan(x/z)
    alpha = rot_y - rot_view
    return (h, w, l, x, y, z, rot_y, alpha)

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2.T


def estimate_ground_plane(annos_3d, camera, frameId):
    avg_dir_vector = np.array([0.0, 0.0, 0.0])

    for anno3d in annos_3d:
        vertices = anno3d.vertices
    
        # curr_pose = camera.cam2world[frameId]
        curr_pose = get_camera_pose(camera, frameId)
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        points_local = camera.world2cam(vertices, R, T, inverse=True)

        assert points_local.shape[1] == 8
        # points_local = camera.world2cam(points_local, R, T, inverse=True)
        points_local = np.transpose(points_local)

        avg_dir_vector += np.absolute(points_local[3] - points_local[6])
        avg_dir_vector += np.absolute(points_local[2] - points_local[7])
        avg_dir_vector += np.absolute(points_local[0] - points_local[5])
        avg_dir_vector += np.absolute(points_local[1] - points_local[4])
        avg_dir_vector[0] = 0.0

    avg_dir_vector = avg_dir_vector / np.linalg.norm(avg_dir_vector)
    rotation_x = -1*np.arctan(avg_dir_vector[1] / avg_dir_vector[2])

    rot_matrix = np.array([[1, 0, 0], [0, np.cos(rotation_x), -1*np.sin(rotation_x)], [0, np.sin(rotation_x), np.cos(rotation_x)]])
    return rot_matrix


def get_kitti_annotations(anno3d, camera, sequence, frameId, ground_plane=None, return_str=True):
    """Converts the 3D bounding boxes to KITTI format given the annotation object"""

    vertices = anno3d.vertices
    
    # curr_pose = camera.cam2world[frameId]
    curr_pose = get_camera_pose(camera, frameId)
    T = curr_pose[:3,  3]
    R = curr_pose[:3, :3]

    points_local = camera.world2cam(vertices, R, T, inverse=True)

    assert points_local.shape[1] == 8

    # type
    type = KITTI_CATEGORIES[id2label[anno3d.semanticId].name]

    if ground_plane is not None:
        points_local = np.transpose(np.matmul(ground_plane, points_local))
    else:
        points_local = np.transpose(points_local)
        # points_local = remove_pitch(points_local)
    
    # dimensions 
    # points_local = np.transpose(points_local)

    # points_bbox = corrected_bbox(points_local)
    points_bbox = points_local
    h, w, l, x, y, z, rot_y, alpha = get_bbox_atttribute(points_bbox)

    # if the z is larger than threshold, then ignore
    if z > MAX_BOX_DIST:
        # print("Box exceeds max distance")
        return ''
    
    ## Test
    # import vedo
    # points_proj = compute_3d_box_cam2(h, w, l, x, y, z, rot_y)
    # pts_0 = vedo.Points(points_local, r=10, c='g')
    # pts_1 = vedo.Points(points_bbox, r=10, c='r')
    # pts_2 = vedo.Points(points_proj, r=10, c='b')
    # vedo.show([pts_0, pts_1, pts_2], axes=1)
    # quit()

    # points_local = remove_pitch(points_local)

    # height = np.linalg.norm((points_local[2] - points_local[3]))
    # width = np.linalg.norm((points_local[1] - points_local[3]))
    # length = np.linalg.norm((points_local[3] - points_local[6]))

    # dims = [height, width, length]

    # # location 
    # center = (points_local[1] + points_local[3] + points_local[4] + points_local[6]) / 4
    # location = center

    # # rotation_y 
    # vec = (points_local[3] - points_local[6])
    # rotation_y = -1*np.arctan2(vec[2], vec[0])

    # # alpha 
    # alpha = rotation_y - np.arctan(vec[0] / vec[2])

    # bbox 
    instance_file = get_instance_map_path(sequence, frameId)
    instance_seg = io.imread(instance_file)

    global_id = anno3d.semanticId * MAX_N + anno3d.instanceId
    
    # print("instane seg shape: ", instance_seg.shape)
    rows, cols = np.where(instance_seg == global_id)

    u_min, v_min, u_max, v_max = np.min(rows), np.min(cols), np.max(rows), np.max(cols)
    bbox_2d = [u_min, v_min, u_max, v_max]

    # ==============================================================================
    # Truncation
    # ============================================================================== 
    
    points_bbox = np.transpose(points_bbox)
    # u, v, depth = camera.cam2image(np.matmul(np.linalg.inv(ground_plane), points_bbox))
    u, v, depth = camera.cam2image(points_bbox)

    u_min, u_max = np.min(u), np.max(u)
    v_min, v_max = np.min(v), np.max(v)
    box3d_area = (u_max - u_min) * (v_max - v_min)

    u_min_in = np.clip(u_min, 0, IMAGE_W)
    u_max_in = np.clip(u_max, 0, IMAGE_W)
    v_min_in = np.clip(v_min, 0, IMAGE_H)
    v_max_in = np.clip(v_max, 0, IMAGE_H)
    inside_area = (u_max_in - u_min_in) * (v_max_in - v_min_in)
    truncation = 1 - (inside_area / box3d_area)

    # img = instance_seg == global_id
    # plt.imshow(img)
    # plt.show()
    # ==============================================================================
    # Occlusion
    # ============================================================================== 
    globalId_cnt = np.sum(instance_seg == global_id)
    visible_frac   = globalId_cnt/box3d_area if box3d_area > 0 else 0.0

    VISIBLE_FRAC_THRESHOLD_1 = 0.6
    VISIBLE_FRAC_THRESHOLD_2 = 0.2

    if visible_frac > VISIBLE_FRAC_THRESHOLD_1:
        occlusion = 0
    elif visible_frac > VISIBLE_FRAC_THRESHOLD_2:
        occlusion = 1
    else:
        occlusion = 2

    name = type
    name += ' '
    trunc = '{:.2f} '.format(truncation)
    occ = '{:d} '.format(occlusion)
    a = '{:.2f} '.format(alpha)
    bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
    hwl = '{:.2} {:.2f} {:.2f} '.format(h, w, l)  # height, width, length.
    xyz = '{:.2f} {:.2f} {:.2f} '.format(x, y, z)  # x, y, z.
    y = '{:.2f}'.format(rot_y)  # Yaw angle.
    # s = ' {:.4f}'.format(box.score)  # Classification score.

    output = name + trunc + occ + a + bb + hwl + xyz + y
    
    return output


def test():
    from label_helpers import Annotation3D
    from kitti_helper import get_kitti_annotations
    from project import CameraPerspective
    from convert_data import create_instance_3d_dict, check_for_instance_segmentation, get_instance_ids, get_annos_3d

    seq = '2013_05_28_drive_0000_sync'
    frame_id = 1540
    camera = CameraPerspective(ROOT_DIR, seq, CAM_ID)

    label3DBboxPath = os.path.join(ROOT_DIR, 'data_3d_bboxes/train')
    pose_dir = os.path.join(ROOT_DIR, POSES_DATA)
    annotation3D = Annotation3D(label3DBboxPath, seq, posesDir=pose_dir)
    instance_3d_dict = create_instance_3d_dict(annotation3D)

    cam = 'image_%02d' % CAM_ID + '/data_rect/'
    all_imgs = sorted(os.listdir(os.path.join(ROOT_DIR, RAW_IMAGE_DATA, seq, cam)))

    img = all_imgs[frame_id]
    print('Image: {}'.format(img))
    img_name = img[:-4]
    frame = int(img_name)

    if not check_for_instance_segmentation(seq, frame):
        print('[Error] No segmentation label')
        quit()

    instance_ids = get_instance_ids(seq, frame)
    annos_3d = get_annos_3d(instance_3d_dict, instance_ids)
    if len(annos_3d) == 0:  # make sure we have 3d annotations
        print('[Error] Length of annos_3d == 0')
        quit()
    
    for anno in annos_3d:
        output = get_kitti_annotations(anno, camera, seq, frame)
        if output == '':
            continue
        print(output)

if __name__ == '__main__':
    test()