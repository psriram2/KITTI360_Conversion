import os
import numpy as np
import re
import yaml
import sys
from calib_helpers import loadCalibrationCameraToPose

from skimage import io
from labels import id2label

MAX_N = 1000

def readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

class Camera:
    def __init__(self):
        
        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            # consider the rectification for perspective cameras
            if self.cam_id==0 or self.cam_id==1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id==2 or self.cam_id==3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')


    def world2cam(self, points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        if ndim==2:
            points = points[0]

        return points

    def cam2image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError
    
    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u,v,depth = self.cam2image(points_local)

        return (u,v), depth 

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 
        obj3d.generateMeshes()


class CameraPerspective(Camera):

    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=0):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==0 or cam_id==1)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraPerspective, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load perspective intrinsics '''
    
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % self.cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % self.cam_id:
                R_rect = np.eye(4) 
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_%02d:" % self.cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)
    
        self.K = K
        self.width, self.height = width, height
        self.R_rect = R_rect

        self.temp_K = None

    def cam2image(self, points, temp_K=False):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        
        if not temp_K:
            points_proj = np.matmul(self.K[:3,:3].reshape([1,3,3]), points)
        else:
            points_proj = np.matmul(self.temp_K[:3,:3].reshape([1,3,3]), points)

        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int32)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int32)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

class CameraFisheye(Camera):
    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=2):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==2 or cam_id==3)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'image_%02d.yaml' % cam_id)
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraFisheye, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load fisheye intrinsics '''

        intrinsics = readYAMLFile(intrinsic_file)

        self.width, self.height = intrinsics['image_width'], intrinsics['image_height']
        self.fi = intrinsics

    def cam2image(self, points):
        ''' camera coordinate to image plane '''
        points = points.T
        norm = np.linalg.norm(points, axis=1)

        x = points[:,0] / norm
        y = points[:,1] / norm
        z = points[:,2] / norm

        x /= z+self.fi['mirror_parameters']['xi']
        y /= z+self.fi['mirror_parameters']['xi']

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        ro2 = x*x + y*y
        x *= 1 + k1*ro2 + k2*ro2*ro2
        y *= 1 + k1*ro2 + k2*ro2*ro2

        x = gamma1*x + u0
        y = gamma2*y + v0

        return x, y, norm * points[:,2] / np.abs(points[:,2])

def get_new_points_local(points_local):
    
    points_local_copy = np.copy(points_local)

    new_points_local = np.array([[], [], [], [], [], [], [], []])

    min_x = np.min(points_local[:, 0])
    min_y = np.min(points_local[:, 1])
    min_z = np.min(points_local[:, 2])
    max_x = np.max(points_local[:, 0])
    max_y = np.max(points_local[:, 1])
    max_z = np.max(points_local[:, 2])


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

    print("unit_l_dir: ", unit_l_dir)
    print("unit_h_dir: ", unit_h_dir)
    print("unit_w_dir: ", unit_w_dir)

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
    # w_min_dist = np.min(w_points)*unit_w_dir
    # w_max_dist = np.max(w_points)*unit_w_dir

    
    proj_points[2] = proj_points[2] + (w_points[w_min]*unit_w_dir - w_points[2]*unit_w_dir)
    proj_points[3] = proj_points[3] + (w_points[w_min]*unit_w_dir - w_points[3]*unit_w_dir)
    proj_points[6] = proj_points[6] + (w_points[w_min]*unit_w_dir - w_points[6]*unit_w_dir)
    proj_points[7] = proj_points[7] + (w_points[w_min]*unit_w_dir - w_points[7]*unit_w_dir)


    proj_points[0] = proj_points[0] + (w_points[w_max]*unit_w_dir - w_points[0]*unit_w_dir)
    proj_points[1] = proj_points[1] + (w_points[w_max]*unit_w_dir - w_points[1]*unit_w_dir)
    proj_points[4] = proj_points[4] + (w_points[w_max]*unit_w_dir - w_points[4]*unit_w_dir)
    proj_points[5] = proj_points[5] + (w_points[w_max]*unit_w_dir - w_points[5]*unit_w_dir)



    print("proj points: ", proj_points)

    print("l_min: ", l_min)
    print("l_max: ", l_max)
    print("w_min: ", w_min)
    print("w_max: ", w_max)

    return proj_points


def get_new_points_local2(points_local):

    from numpy import array,mat,sin,cos,dot,eye
    from numpy.linalg import norm

    def rodrigues(r):
        def S(n):
            Sn = array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
            return Sn
        theta = norm(r)
        if theta > 1e-30:
            n = r/theta
            Sn = S(n)
            R = eye(3) + sin(theta)*Sn + (1-cos(theta))*dot(Sn,Sn)
        else:
            Sr = S(r)
            theta2 = theta**2
            R = eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*dot(Sr,Sr)
        print("R shape: ", R.shape)
        # return array(R)
        return R

    # perform pitch warping
    new_points_local = []
    print("points local: ", points_local)
    
    vec = (points_local[3] - points_local[6])
    rotation_x = np.arctan(-1*vec[1]/(np.sqrt(vec[0]**2 + vec[2]**2))) # could be negative

    proj_xz = np.array([vec[0], 0.0, vec[2]])
    proj_xz = proj_xz / np.linalg.norm(proj_xz)

    unit_h_dir = np.array([0, 1, 0])
    rot_axis = np.cross(proj_xz, unit_h_dir)
    print("rot_axis: ", rot_axis)

    rot_matrix = rodrigues(rotation_x * rot_axis)
    print("rot_matrix: ", rot_matrix)
    
    new_points_local = rot_matrix @ np.transpose(points_local)
    # 1/0

    return np.transpose(new_points_local), np.linalg.inv(rot_matrix)
    pass

def estimate_ground_plane(annotation3D):
    print("ESTIMATE GROUND PLANE: ", frame, instanceIDs)

    avg_dir_vector = np.array([0.0, 0.0, 0.0])

    for k,v in annotation3D.objects.items():
        if len(v.keys())==1 and (-1 in v.keys()): # show static only
            obj3d = v[-1]
            if not id2label[obj3d.semanticId].name=='car': # show buildings only
                continue

            if obj3d.semanticId * MAX_N + obj3d.instanceId not in instanceIDs:
                continue 

            # print("obj3d vertices: ", obj3d.vertices)
            
            camera(obj3d, frame)
            # vertices = np.asarray(obj3d.vertices_proj).T
            # points.append(np.asarray(obj3d.vertices_proj).T)
            # depths.append(np.asarray(obj3d.vertices_depth))


            points_local = obj3d.vertices
            curr_pose = camera.cam2world[frame]
            T = curr_pose[:3,  3]
            R = curr_pose[:3, :3]

            # print("world corodinates: ", points_local)

            # convert points from world coordinate to camera coordinate 
            points_local = camera.world2cam(points_local, R, T, inverse=True)
            points_local = np.transpose(points_local)


            # print("points local: ", points_local)
            # avg_dir_vector += (points_local[3] - points_local[6])
            # avg_dir_vector += (points_local[2] - points_local[7])
            # avg_dir_vector += (points_local[0] - points_local[5])
            # avg_dir_vector += (points_local[1] - points_local[4])
            avg_dir_vector += np.absolute(points_local[3] - points_local[6])
            avg_dir_vector += np.absolute(points_local[2] - points_local[7])
            avg_dir_vector += np.absolute(points_local[0] - points_local[5])
            avg_dir_vector += np.absolute(points_local[1] - points_local[4])
            avg_dir_vector[0] = 0.0

    avg_dir_vector = avg_dir_vector / np.linalg.norm(avg_dir_vector)
    rotation_x = -1*np.arctan(avg_dir_vector[1] / avg_dir_vector[2])

    rot_matrix = np.array([[1, 0, 0], [0, np.cos(rotation_x), -1*np.sin(rotation_x)], [0, np.sin(rotation_x), np.cos(rotation_x)]])
    print("ground plane estimation done.")
    return rot_matrix

if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt
    from labels import id2label

    # if 'KITTI360_DATASET' in os.environ:
    #     kitti360Path = os.environ['KITTI360_DATASET']
    # else:
    #     kitti360Path = os.path.join(os.path.dirname(
    #                             os.path.realpath(__file__)), '..', '..')
    
    kitti360Path = "../KITTI360_scripts/"
    poses_data = "./data_poses"

    seq = 0
    cam_id = 0
    sequence = '2013_05_28_drive_%04d_sync'%seq
    # perspective
    if cam_id == 0 or cam_id == 1:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
        print(camera.fi)
    else:
        raise RuntimeError('Invalid Camera ID!')


    # 3D bbox
    from label_helpers import Annotation3D
    label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes/train')
    annotation3D = Annotation3D(label3DBboxPath, sequence, posesDir=poses_data)


    # print("camera frames: ", camera.frames)
    # 1/0

    # loop over frames
    for frame in camera.frames:

        # # REMOVE !!
        # if frame != 791:
        #     print("Frame: ", frame)
        #     continue

        # perspective
        if cam_id == 0 or cam_id == 1:
            image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect', '%010d.png'%frame)
            instance_file = os.path.join(kitti360Path, 'data_2d_semantics/train', sequence, 'image_%02d' % cam_id, 'instance', '%010d.png'%frame)
        # fisheye
        elif cam_id == 2 or cam_id == 3:
            image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb', '%010d.png'%frame)
        else:
            raise RuntimeError('Invalid Camera ID!')
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue


        if not os.path.exists(instance_file):
            continue

        instance_seg = io.imread(instance_file)

        # print("instance seg: ", instance_seg.shape)
        instanceIDs = set()
        # semanticIDs = set()

        nrows, ncols = instance_seg.shape
        for i in range(nrows):
            for j in range(ncols):
                instanceIDs.add(instance_seg[i][j])
                # semanticIDs.add(id2label[instance_seg[i][j] // 1000].category)
        
        # print("instanceIds: ", instanceIDs)
        # print("semantic names: ", semanticIDs)
        # 1/0

        gplane = estimate_ground_plane(annotation3D)
            

        # print(image_file)
        image = cv2.imread(image_file)
        plt.imshow(image[:,:,::-1])

        points = []
        depths = []
        for k,v in annotation3D.objects.items():
            if len(v.keys())==1 and (-1 in v.keys()): # show static only
                obj3d = v[-1]
                if not id2label[obj3d.semanticId].name=='car': # show buildings only
                    continue

                if obj3d.semanticId * MAX_N + obj3d.instanceId not in instanceIDs:
                    continue 

                # print("obj3d vertices: ", obj3d.vertices)
                
                camera(obj3d, frame)
                vertices = np.asarray(obj3d.vertices_proj).T
                points.append(np.asarray(obj3d.vertices_proj).T)
                depths.append(np.asarray(obj3d.vertices_depth))


                points_local = obj3d.vertices
                curr_pose = camera.cam2world[frame]
                T = curr_pose[:3,  3]
                R = curr_pose[:3, :3]

                # print("world corodinates: ", points_local)

                # convert points from world coordinate to camera coordinate 
                points_local = camera.world2cam(points_local, R, T, inverse=True)
                points_local = np.transpose(points_local)


                # print("points local: ", points_local)


                # new_points_local = get_new_points_local(points_local)
                # points_local = get_new_points_local(points_local)
                # points_local, inv_rot = get_new_points_local2(points_local)

                points_local = np.transpose(gplane @ np.transpose(points_local))
                
                # if frame == 791:
                #     print("points_local: ", points_local)
                #     # 1/0
                # else:
                #     print(frame)


                inv_rot = np.linalg.inv(gplane)

                # print("points local shape: ", points_local.shape)

                # print("new points local: ", points_local)
                # 1/0


                # print("new points local: ", points_local)

                height = np.linalg.norm((points_local[2] - points_local[3]))
                width = np.linalg.norm((points_local[1] - points_local[3]))
                length = np.linalg.norm((points_local[3] - points_local[6]))
                # print("hwl : ", height, width ,length)

                # height = max(np.linalg.norm((points_local[2] - points_local[3])), np.linalg.norm((points_local[0] - points_local[1])), np.linalg.norm((points_local[7] - points_local[6])), np.linalg.norm((points_local[5] - points_local[4])))
                # width = max(np.linalg.norm((points_local[1] - points_local[3])), np.linalg.norm((points_local[0] - points_local[2])), np.linalg.norm((points_local[5] - points_local[7])), np.linalg.norm((points_local[4] - points_local[6])))
                # length = max(np.linalg.norm((points_local[3] - points_local[6])), np.linalg.norm((points_local[2] - points_local[7])), np.linalg.norm((points_local[0] - points_local[5])), np.linalg.norm((points_local[1] - points_local[4])))
                

                # print("height: ", (points_local[2] - points_local[3]))
                # print("width: ", (points_local[1] - points_local[3]))
                # print("length: ", (points_local[3] - points_local[6]))

                # height = np.sqrt(((points_local[2] - points_local[3])[1])**2)
                # width = np.sqrt(((points_local[1] - points_local[3])[0])**2)
                # length = np.sqrt(((points_local[3] - points_local[6])[2])**2)

                

                # print("min x: ", min_x)
                # height = np.sqrt((min_y - max_y)**2)
                # width = np.sqrt((min_x - max_x)**2)
                # length = np.sqrt((min_z - max_z)**2)

                # print("new hwl : ", height, width ,length)

                # points_local = []

                # print("local hwl: ", obj3d.obj_height, obj3d.obj_width, obj3d.obj_length)

                # location 
                
                center = (points_local[1] + points_local[3] + points_local[4] + points_local[6]) / 4
                
                # center[1] = max(points_local[1, 1], points_local[3, 1], points_local[4, 1], points_local[6, 1])
                # center = np.array([(min_x + max_x)/2, (min_y + max_y) / 2, (min_z + max_z)/ 2])
                # print("center: ", center)

                # print(obj3d.bbox_center.shape)
                # center_local = camera.world2cam(np.reshape(obj3d.bbox_center, (1, 3)), R, T, inverse=True)
                # print("center local: ", center_local)
                # 1/0

                # rotation_y 
                # print("points local shape: ", points_local.shape)
                vec = (points_local[3] - points_local[6])
                # print("vec: ", vec_xz[0].shape)
                # rotation_y = -1*np.arctan2(vec[2], vec[0])
                rotation_y = -1*np.arctan2(vec[2], vec[0])
                # print("rotation_y: ", rotation_y*180 / np.pi)

                # alpha 
                alpha = rotation_y - np.arctan(vec[0] / vec[2])
                # print("alpha: ", alpha * 180 / np.pi)
                # 1/0

                dims = [height, width, length]
                rot_y = rotation_y

                # print("everyting: ", rot_y, dims, center, alpha)

                box_3d = []
                for i in [1,-1]:
                    for j in [1,-1]:
                        for k in [0, 1]:
                            point = np.copy(center)
                            point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                            point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
                            point[1] = center[1] - k * dims[0]

                            # point = np.append(point, 1)
                            # point = np.dot(cam_to_img, point)
                            # point = point[:2]/point[2]
                            # point = point.astype(np.int16)
                            box_3d.append(point)


                # reordered_box_3d = [box_3d[5], box_3d[4], box_3d[1], box_3d[0], box_3d[6], box_3d[7], box_3d[2], box_3d[3]]
                reordered_box_3d = [box_3d[1], box_3d[0], box_3d[7], box_3d[6], box_3d[2], box_3d[3], box_3d[4], box_3d[5]]
                # obj3d.vertices = np.array(reordered_box_3d)
                # obj3d.vertices = points_local

                new_vertices = np.array(reordered_box_3d)

                # print("new vertices (cam coordinates): ", new_vertices)

                # new_vertices = np.transpose(inv_rot @ np.transpose(new_vertices))
                camera.temp_K = np.zeros((3, 4))
                camera.temp_K[:3, :3] = np.matmul(camera.K[:3, :3], inv_rot)

                print("DIFFERENCE: ", new_vertices - points_local)
                print("#"*100)
                
                # print("ground plane: ", gplane)
                # print("orig K: ", camera.K)
                # print("result: ",  np.matmul(camera.K[:3, :3], np.linalg.inv(gplane)))
                # print("K: ", camera.temp_K)
                # 1/0





                for line in obj3d.lines:
                # for line in [obj3d.lines[3]]:
                # for line in [[3, 6]]:
                    # v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                    # v = []
                    # center = (obj3d.vertices[1] + obj3d.vertices[3] + obj3d.vertices[4] + obj3d.vertices[6]) / 4
                    # print("real center: ", center)
                    # v.append(center)
                    # v.append(new_vertices[0])
                    # print("v: ", v)
                    # v = [new_vertices[5]]

                    v = [new_vertices[line[0]]*x + new_vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                    # v = [points_local[line[0]]*x + points_local[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                    u, v, depth = camera.cam2image(np.transpose(v), temp_K=True)
                    uv, d = (u, v), depth

                    # uv, d = camera.project_vertices(np.asarray(v), frame)
                    mask = np.logical_and(np.logical_and(d>0, uv[0]>0), uv[1]>0)
                    mask = np.logical_and(np.logical_and(mask, uv[0]<image.shape[1]), uv[1]<image.shape[0])
                    plt.plot(uv[0][mask], uv[1][mask], 'r.', linewidth=0.1, markersize=1)

        plt.pause(0.5)
        plt.clf()
        # 1/0