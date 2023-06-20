

# f"/home/psriram2/pranav/kitti360/KITTI360_Conversion/image_2_orig/2013_05_28_drive_0000_sync_CAM0_{}.png"


import shutil 
import os 

DIR_PATH = "/home/psriram2/pranav/kitti360/KITTI360_Conversion/image_2_allsmog/"
ORIG_PATH = "/home/psriram2/pranav/kitti360/KITTI360_Conversion/image_2_orig/"
NEW_PATH = "/home/psriram2/pranav/kitti360/KITTI360_Conversion/image_2/"
present_files = set([f for f in os.listdir(ORIG_PATH)])

cnt = 0
for f in os.listdir(DIR_PATH):
    if f.endswith(".png") and f"2013_05_28_drive_0000_sync_CAM0_{f}" in present_files:
        src_dir = DIR_PATH + f
        dst_dir = NEW_PATH + f
        shutil.copy(src_dir,dst_dir)
        cnt += 1

print("cnt: ", cnt)