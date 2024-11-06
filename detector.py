import torch
from mmdet3d.apis import inference_detector, init_model
from mmengine.config import Config
from torch.cuda.amp import autocast
import numpy as np

class Detector:
    def __init__(self):
        # Paths to config and checkpoint files
        self.config_file = '/home/tsr/CARNATIONS/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        self.checkpoint_file = '/home/tsr/CARNATIONS/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'      

	 # Load configuration file and model
        config = Config.fromfile(self.config_file)
        self.model = init_model(config, checkpoint=self.checkpoint_file, device='cuda:0')
        
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
        
 	# Retrieve LiDAR data
        lidar_frame_id, lidar_data = sensor_data.get('LIDAR', (None, None))

        if lidar_data is None:
            print("[ERROR] No LiDAR data found.")
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        # Convert lidar data to torch tensor and move to GPU if needed
        lidar_data_tensor = torch.tensor(lidar_data, dtype=torch.float32).to('cuda:0')

        # Run inference
        try:
            detection_results, _ = inference_detector(self.model, lidar_data_tensor)
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}")
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        # Extract prediction instances
        prediction_instances = detection_results.pred_instances_3d

        # Check if there are any detected instances
        if len(prediction_instances.bboxes_3d) == 0:
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        # Extract labels, scores, and bounding boxes
        detected_labels = prediction_instances.labels_3d.detach().cpu().numpy().reshape(-1)
        detected_scores = prediction_instances.scores_3d.detach().cpu().numpy().reshape(-1)
        bounding_boxes_3d = prediction_instances.bboxes_3d

        # Sort scores in descending order and get integer indices
        sorted_indices = np.argsort(-detected_scores)
        detected_labels = detected_labels[sorted_indices]
        detected_scores = detected_scores[sorted_indices]
        bounding_boxes_3d = bounding_boxes_3d[sorted_indices]

        # Initialize the array for storing corner points of bounding boxes
        num_objects = len(bounding_boxes_3d)
        corner_points_array = np.zeros((num_objects, 8, 3))

        # Calculate the corner points of each bounding box
        for idx in range(num_objects):
            box_params = bounding_boxes_3d[idx].cpu().numpy().flatten()
            x, y, z, length, width, height, orientation = box_params

            # Calculate the corner points
            corner_points_array[idx] = [
                [x, y, z],
                [x, y + width, z],
                [x + length, y + width, z],
                [x + length, y, z],
                [x, y, z + height],
                [x, y + width, z + height],
                [x + length, y + width, z + height],
                [x + length, y, z + height]
            ]

        # Return the sorted detection results
        return {
            "det_class": detected_labels.reshape(-1, 1),
            "det_score": detected_scores.reshape(-1, 1),
            "det_boxes": corner_points_array
        }
