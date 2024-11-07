from mmdet3d.apis import init_model, inference_detector
import numpy as np
import math

class Detector:
    def __init__(self):
        # Paths to the configuration and checkpoint files
        config_path = '/home/tsr/CARNATIONS/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        checkpoint_path = '/home/tsr/CARNATIONS/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        
        # Initialize the model with the provided configuration and checkpoint
        self.model = init_model(config_path, checkpoint_path, device='cuda:0')

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensors required for object detection, positioned relative to the vehicle center.
        :return: A list containing specifications for each sensor.
        """
        sensors = [
            # Front camera for visual data - capture the forward view
            {'type': 'sensor.camera.rgb', 'x': 1.5, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'FrontCamera'},

	    # Left RGB Camera - covers the left side of the vehicle
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            # Right RGB Camera - covers the right side of the vehicle
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            # Rear RGB Camera - captures the view behind the vehicle
            {'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
             'width': 1280,'height': 720, 'fov': 100, 'id': 'Rear'},

            # LiDAR sensor mounted on top for 360-degree coverage
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20,
             'points_per_second': 2304000, 'id': 'TopLiDAR'},

            # GPS sensor for precise positioning
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},

            # IMU sensor for tracking vehicle dynamics
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'id': 'IMU'}
        ]
        return sensors

    def detect(self, sensor_data):
        """
        Perform object detection using sensor data.
        
        :param sensor_data: Dictionary with sensor data. Keys are sensor IDs and values are (frame_id, data) tuples.
        :return: Dictionary with detection results including bounding boxes, object classes, and confidence scores.
        """
        
        # Extract LiDAR data
        _, lidar_data = sensor_data.get("TopLiDAR", (None, None))
        if lidar_data is None:
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        # Transform LiDAR coordinates to match the coordinate system expected by the model
        lidar_data[:, [0, 1]] = lidar_data[:, [1, 0]]  # Swap x and y axes to match model convention
        lidar_data[:, 1] = -lidar_data[:, 1]  # Invert y axis to match right-handed coordinate system

        # Run inference using the provided LiDAR data
        detection_results, _ = inference_detector(self.model, lidar_data)
        predictions = detection_results.pred_instances_3d

        # Extract detected labels, scores, and bounding boxes
        labels = predictions.labels_3d.detach().cpu().numpy().reshape(-1, 1)
        scores = predictions.scores_3d.detach().cpu().numpy().flatten()  # Flatten to ensure 1D array
        bounding_boxes = predictions.bboxes_3d

        num_objects = bounding_boxes.shape[0]
        if num_objects == 0:
            print("[INFO] No objects detected.")
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        box_corners = np.zeros((num_objects, 8, 3))

        # Compute the bounding box corner points for each detected object
        for i in range(num_objects):
            box_params = bounding_boxes[i].detach().cpu().numpy().flatten()
            center_x, center_y, center_z, length, width, height, yaw_angle = box_params

            # Adjust coordinates to align with CARLA's coordinate system
            rotation_matrix = np.array([
                [math.cos(yaw_angle), -math.sin(yaw_angle)],
                [math.sin(yaw_angle), math.cos(yaw_angle)]
            ])

            # Define corner offsets
            offsets = np.array([
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
                [length / 2, width / 2],
                [length / 2, -width / 2]
            ])

            # Rotate and translate corner points
            rotated_offsets = np.dot(offsets, rotation_matrix.T)
            for j in range(4):
                box_corners[i, j, :2] = [center_x + rotated_offsets[j, 0], center_y + rotated_offsets[j, 1]]
                box_corners[i, j, 2] = center_z
                box_corners[i, j + 4, :2] = [center_x + rotated_offsets[j, 0], center_y + rotated_offsets[j, 1]]
                box_corners[i, j + 4, 2] = center_z + height

        # Debugging: Print detected scores and box corners to verify correctness
        print(f"[DEBUG] Detected scores: {scores}")
        print(f"[DEBUG] Detected box corners: {box_corners}")

        # Set a lower confidence threshold for visualization
        confidence_threshold = 0.1
        valid_indices = scores > confidence_threshold

        if not any(valid_indices):
            print("[INFO] No detections above confidence threshold.")
            return {
                "det_class": np.array([]),
                "det_score": np.array([]),
                "det_boxes": np.zeros((0, 8, 3))
            }

        # Filter boxes, labels, and scores by confidence threshold
        box_corners = box_corners[valid_indices]
        labels = labels[valid_indices]
        scores = scores[valid_indices]

        # Return the detection results
        return {
            "det_class": labels,
            "det_score": scores,  # Keep as a flattened 1D array
            "det_boxes": box_corners  # Provide in (N, 8, 3) format for 3D visualization
        }

