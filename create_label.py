import os 
import numpy as np

from label_helpers import Annotation3D


# def loadBoundingBoxes():
#     for globalId,v in self.annotation3D.objects.items():
#         # skip dynamic objects
#         if len(v)>1:
#             continue
#         for obj in v.values():
#             lines=np.array(obj.lines)
#             vertices=obj.vertices
#             faces=obj.faces
#             mesh = open3d.geometry.TriangleMesh()
#             mesh.vertices = open3d.utility.Vector3dVector(obj.vertices)
#             mesh.triangles = open3d.utility.Vector3iVector(obj.faces)
#             color = self.assignColor(globalId, 'semantic')
#             semanticId, instanceId = global2local(globalId)
#             mesh.paint_uniform_color(color.flatten())
#             mesh.compute_vertex_normals()
#             self.bboxes.append( mesh )
#             self.bboxes_window.append([obj.start_frame, obj.end_frame])


raw_data = "./data_2d_raw"
gt_label_data = "./data_3d_bboxes/train"
poses_data = "./data_poses"

label_folder = "./label_2"
if not os.path.exists(label_folder):
   os.makedirs(label_folder)


# example path: data_2d_raw/2013_05_28_drive_{seq:0>4}_sync/image_{00|01}/data_rect/{frame:0>10}.png
for seq in os.listdir(raw_data):
    if seq == ".DS_Store":
        continue

    cam = "image_00/data_rect/"
    bboxes_3d_labels = os.path.join(gt_label_data, seq + ".xml")
    annotation3D = Annotation3D(labelDir=gt_label_data, sequence=seq, posesDir=poses_data)

    1/0

    for img in os.listdir(os.path.join(raw_data, seq, cam)):
        img_name = img[:-4]
        print(f"processing image {img_name}.")

        label_path = os.path.join(label_folder, img_name + '.txt')
        



        
    for sample_annotation_token in sample_annotation_tokens:
        sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

        # Get box in LIDAR frame.
        _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                            selected_anntokens=[sample_annotation_token])
        box_lidar_nusc = box_lidar_nusc[0]

        # Truncated: Set all objects to 0 which means untruncated.
        truncated = 0.0

        # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
        occluded = 0

        # Convert nuScenes category to nuScenes detection challenge category.
        detection_name = category_to_detection_name(sample_annotation['category_name'])

        # Skip categories that are not part of the nuScenes detection challenge.
        if detection_name is None:
            continue

        # Convert from nuScenes to KITTI box format.
        box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
            box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

        # Project 3d box to 2d box in image, ignore box if it does not fall inside.
        bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
        if bbox_2d is None:
            continue

        # Set dummy score so we can use this file as result.
        box_cam_kitti.score = 0

        # Convert box to output string format.
        output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                        truncation=truncated, occlusion=occluded)

        # Write to disk.
        label_file.write(output + '\n')
