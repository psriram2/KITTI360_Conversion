import os
import numpy as np

from skimage import io
from tqdm import tqdm

# local imports
from label_helpers import Annotation3D
from labels import id2label
from kitti_helper import get_kitti_annotations, estimate_ground_plane
from project import CameraPerspective
from calib_helpers import loadPerspectiveIntrinsic, loadCalibrationRigid, loadCalibrationCameraToPose
from config import config
from convert_data import get_instance_map_path, check_for_instance_segmentation, get_instance_ids, create_instance_3d_dict, get_annos_3d


ROOT_DIR = config["ROOT_DIR"]
RAW_IMAGE_DATA = config["RAW_IMAGE_DATA"]
INSTANCE_SEG_DATA = config["INSTANCE_SEG_DATA"]
POSES_DATA = config["POSES_DATA"]
GT_CALIB_DATA = config["GT_CALIB_DATA"]

KITTI_IMAGE_FOLDER = config["KITTI_IMAGE_FOLDER"]
KITTI_LABEL_FOLDER = config["KITTI_LABEL_FOLDER"]
KITTI_CALIB_FOLDER = config["KITTI_CALIB_FOLDER"]
KITTI_SUBSEQUENCE_FOLDER = config["KITTI_SUBSEQUENCE_FOLDER"]

CAM_ID = config["CAM_ID"]
SEQUENCE = config["SEQUENCE"]
CATEGORIES = config["CATEGORIES"]
MAX_N = config["MAX_N"]
SUBSAMPLE_SIZE = config["SUBSAMPLE_SIZE"]
NUM_SUBSEQUENCES = config["NUM_SUBSEQUENCES"]


def subsample(sequence):
    # print(f"processing sequence {sequence} ({seq_idx}/{len(SEQUENCE)})")
    seq = '2013_05_28_drive_{:0>4d}_sync'.format(sequence)
    camera = CameraPerspective(ROOT_DIR, seq, CAM_ID)

    label3DBboxPath = os.path.join(ROOT_DIR, 'data_3d_bboxes/train')
    pose_dir = os.path.join(ROOT_DIR, POSES_DATA)
    annotation3D = Annotation3D(label3DBboxPath, seq, posesDir=pose_dir)
    instance_3d_dict = create_instance_3d_dict(annotation3D)

    cam = 'image_%02d' % CAM_ID + '/data_rect/'
    all_imgs = os.listdir(os.path.join(ROOT_DIR, RAW_IMAGE_DATA, seq, cam))

    frames = [int(img_name[:-4]) for img_name in all_imgs]
    frames.sort()
    chunks = [frames[i:i + SUBSAMPLE_SIZE] for i in range(0, len(frames), SUBSAMPLE_SIZE)] 
    
    assert len(chunks) > 1

    processed_chunks = []
    
    for chunk_idx in tqdm(range(len(chunks))):
        if chunk_idx >= 8:
            break


        chunk = chunks[chunk_idx]
        total_annos = 0
        pos_annos = 0

        for frame in chunk:

            if not check_for_instance_segmentation(seq, frame):
                # print(f"No instance segmentation found for {frame}")
                continue

            instance_ids = get_instance_ids(seq, frame)
            annos_3d = get_annos_3d(instance_3d_dict, instance_ids)

            # num_annos = 0
            for anno in annos_3d:
                pos_annos += 1
                output = get_kitti_annotations(anno, camera, seq, frame, ground_plane=None)
                if output != '':
                    total_annos += 1

        processed_chunks.append([chunk_idx, total_annos, pos_annos])
        print(f"Chunk {chunk_idx} has {total_annos} annotations, possible annotations: {pos_annos}")

    
    processed_chunks.sort(key=lambda x: x[1], reverse=True)
    sampled_chunks = processed_chunks[:NUM_SUBSEQUENCES]

    subsample_path = os.path.join(KITTI_SUBSEQUENCE_FOLDER, seq + '.txt')
    with open(subsample_path, "w") as subsample_file:

        for i in range(len(sampled_chunks)):
            idx, num_annos = sampled_chunks[i][0], sampled_chunks[i][1]
            pos_annos = sampled_chunks[i][2]
            # print(f"Chunk {idx} has {num_annos} annotations, possible annotations: {pos_annos}")
            
            for frame in chunks[idx]:
                subsample_file.write('{:0>4d}'.format(frame) + f",{idx}" + "\n")




if __name__ == "__main__":

    if not os.path.exists(KITTI_SUBSEQUENCE_FOLDER):
        os.makedirs(KITTI_SUBSEQUENCE_FOLDER, exist_ok=True)
    
    for seq_idx, sequence in enumerate(SEQUENCE):
        print(f"processing sequence {sequence} ({seq_idx}/{len(SEQUENCE)})")
        subsample(sequence)