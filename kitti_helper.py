import os
import numpy as np

from skimage import io

# local imports
from label_helpers import Annotation3D
from labels import id2label
from config import config


ROOT_DIR = config["ROOT_DIR"]
RAW_IMAGE_DATA = config["RAW_IMAGE_DATA"]
INSTANCE_SEG_DATA =  config["INSTANCE_SEG_DATA"]
POSES_DATA = config["POSES_DATA"]

KITTI_LABEL_FOLDER = config["KITTI_LABEL_FOLDER"]

CAM_ID = config["CAM_ID"]
KITTI_CATEGORIES = {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}
MAX_N = config["MAX_N"]


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



def get_kitti_annotations(anno3d, camera, sequence, frameId, return_str=True):
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
    # kitti_annotation["type"] = type
    
    # dimensions 
    points_local = np.transpose(points_local)
    points_local = remove_pitch(points_local)

    height = np.linalg.norm((points_local[2] - points_local[3]))
    width = np.linalg.norm((points_local[1] - points_local[3]))
    length = np.linalg.norm((points_local[3] - points_local[6]))

    dims = [height, width, length]

    # location 
    center = (points_local[1] + points_local[3] + points_local[4] + points_local[6]) / 4
    location = center

    # rotation_y 
    vec = (points_local[3] - points_local[6])
    rotation_y = -1*np.arctan2(vec[2], vec[0])

    # alpha 
    alpha = rotation_y - np.arctan(vec[0] / vec[2])

    # bbox 
    instance_file = get_instance_map_path(sequence, frameId)
    instance_seg = io.imread(instance_file)

    global_id = anno3d.semanticId * MAX_N + anno3d.instanceId
    
    # print("instane seg shape: ", instance_seg.shape)
    rows, cols = np.where(instance_seg == global_id)

    num_pixels = len(rows)
    u_min, v_min, u_max, v_max = np.min(rows), np.min(cols), np.max(rows), np.max(cols)
    bbox_2d = [u_min, v_min, u_max, v_max]
    globalId_cnt = np.sum(instance_seg == global_id)


    # truncated

    # ==============================================================================
    # Truncation
    # ============================================================================== 
    
    points_local = np.transpose(points_local)
    u, v, depth = camera.cam2image(points_local)
    uv_vertices, depth = (u, v) , depth

    u_min_temp = np.min(uv_vertices[0], axis= 0)
    v_min_temp = np.min(uv_vertices[1], axis= 0)
    u_max_temp = np.max(uv_vertices[0], axis= 0)
    v_max_temp = np.max(uv_vertices[1], axis= 0)

    box3d_area = (int(u_max_temp) - int(u_min_temp))*(int(v_max_temp) - int(v_min_temp))

    # # Update projected uv_min if they are larger than min bounds from globalId
    # # projected uv_max if they are smaller than max bounds from globalId
    u_min_temp = u_min if u_min < u_min_temp else u_min_temp
    v_min_temp = v_min if v_min < v_min_temp else v_min_temp
    u_max_temp = u_max if u_max > u_max_temp else u_max_temp
    v_max_temp = v_max if v_max > v_max_temp else v_max_temp

    # # https://github.com/abhi1kumar/groomed_nms/blob/main/data/kitti_split1/devkit/readme.txt#L55-L72
    truncation = 1.0
    if u_min < u_max and v_min < v_max and u_min_temp < u_max_temp and v_min_temp < v_max_temp:
        truncation  = 1.0 - ((u_max - u_min)*(v_max - v_min))/((u_max_temp - u_min_temp)*(v_max_temp - v_min_temp))

    # occlusion
    # box2d_area     = (int(u_max) - int(u_min))*(int(v_max) - int(v_min))
    visible_frac   = globalId_cnt/box3d_area if box3d_area > 0 else 0.0

    VISIBLE_FRAC_THRESHOLD_1 = 0.6
    VISIBLE_FRAC_THRESHOLD_2 = 0.1

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
    hwl = '{:.2} {:.2f} {:.2f} '.format(dims[0], dims[1], dims[2])  # height, width, length.
    xyz = '{:.2f} {:.2f} {:.2f} '.format(location[0], location[1], location[2])  # x, y, z.
    y = '{:.2f}'.format(rotation_y)  # Yaw angle.
    # s = ' {:.4f}'.format(box.score)  # Classification score.

    output = name + trunc + occ + a + bb + hwl + xyz + y
    
    return output


