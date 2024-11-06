import torch
from mmdet3d.apis import inference_detector, init_model
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch.cuda.amp import autocast
import numpy as np
import os

class Detector:
    def __init__(self):
        # Paths to config and checkpoint files
        self.config_file = '/home/tsrlab/mmdetection3d/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py'
        self.checkpoint_file = '/home/tsrlab/CARLA_0.9.15/Edison/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'

        # Check if config and checkpoint exist, for debugging purposes
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_file}")

        # Load configuration file
        config = Config.fromfile(self.config_file)

        # Modify the test pipeline by removing unsupported arguments
        if 'data' in config and 'test' in config['data']:
            pipeline = config['data']['test'].get('pipeline', [])
            for transform in pipeline:
                if 'to_float32' in transform:
                    del transform['to_float32']

        # Initialize the model without loading the checkpoint
        self.model = init_model(config, self.checkpoint_file, device='cuda:0')
        
        # Load the checkpoint separately with strict=False to handle mismatched keys
        load_checkpoint(self.model, self.checkpoint_file, map_location='cuda:0', strict=False)
        
        torch.cuda.empt_cache()

        # pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the detector. The location is defined with respect to the actor center
        -- x axis is longitudinal (forward-backward)
        -- y axis is lateral (left and right)
        -- z axis is vertical
        Unit is in meters

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = [
            # Front RGB Camera - captures the forward view
            {'type': 'sensor.camera.rgb', 'x': 1.5, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Front'},

            # Left RGB Camera - covers the left side of the vehicle
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
            # 'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            # Right RGB Camera - covers the right side of the vehicle
            #{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
            # 'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            # Rear RGB Camera - captures the view behind the vehicle
            #{'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
            # 'width': 1280,'height': 720, 'fov': 100, 'id': 'Rear'},

            # LiDAR - mounted at the top to provide 360-degree point cloud data
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20,
             'points_per_second': 2304000, 'id': 'LIDAR'},

            # GPS Sensor - provides absolute positioning
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},

            # IMU Sensor - captures acceleration and rotation to track vehicle dynamics
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'id': 'IMU'}
        ]
        return sensors

    def detect(self, sensor_data):
        """
        Add your detection logic here
            Input: sensor_data, a dictionary containing all sensor data. Key: sensor id. Value: tuple of frame id and data. For example
                'Right' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'Left' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'LIDAR' : (frame_id, numpy.ndarray)
                    The lidar data, shape (N, 4)
            Output: a dictionary of detected objects in global coordinates
                det_boxes : numpy.ndarray
                    The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
                det_class : numpy.ndarray
                    The object class for each predicted bounding box, shape (N, 1) corresponding to the above bounding box. 
                    0 for vehicle, 1 for pedestrian, 2 for cyclist.
                det_score : numpy.ndarray
                    The confidence score for each predicted bounding box, shape (N, 1) corresponding to the above bounding box.
        """
        # 1. Extract LiDAR and Camera data
        lidar_data = None
        cam_data = None
        for sensor_id, (frame_id, data) in sensor_data.items():
            if sensor_id == 'LIDAR':
                lidar_data = data  # LiDAR point cloud data, shape (N, 4)
            elif sensor_id == 'Front':
                cam_data = data  # Camera RGB data, shape (H, W, 4)

        # 2. Pre-process data for the model
        if lidar_data is None or cam_data is None:
            raise ValueError("Both LiDAR and Camera data are required for detection.")

        # Convert numpy arrays to torch tensors
        lidar_data = torch.from_numpy(lidar_data).float().unsqueeze(0).to('cuda:0')  # (1, N, 4)
        cam_data = torch.from_numpy(cam_data).float().permute(2, 0, 1).unsqueeze(0).to('cuda:0')  # (1, C, H, W)

        # Create input dictionary
        inputs = {
            'points': [lidar_data],  # LiDAR point cloud data
            'imgs': [cam_data]       # Camera RGB image data
        }

        # 3. Run inference
        # *** Add the mixed precision inference here ***
        with autocast():
            result = inference_detector(self.model, inputs)  # Use mixed precision for inference to save GPU memory

        # 4. Post-process results to extract bounding boxes, classes, and scores
        det_boxes, det_class, det_score = [], [], []
        for res in result[0]:  # Assuming detection result is in the format of a list of dictionaries
            boxes = res['boxes_3d'].tensor.cpu().numpy()  # Extract bounding boxes
            scores = res['scores_3d'].cpu().numpy()  # Confidence scores
            labels = res['labels_3d'].cpu().numpy()  # Detected classes

            det_boxes.append(boxes)
            det_score.append(scores)
            det_class.append(labels)

        # Convert lists to numpy arrays
        det_boxes = np.concatenate(det_boxes, axis=0) if det_boxes else np.array([])
        det_class = np.concatenate(det_class, axis=0) if det_class else np.array([])
        det_score = np.concatenate(det_score, axis=0) if det_score else np.array([])

        # Return detection results in the expected format
        return {
            'det_boxes': det_boxes,  # Detected bounding boxes
            'det_class': det_class,  # Object classes (0: vehicle, 1: pedestrian, 2: cyclist)
            'det_score': det_score  # Confidence scores
        }

