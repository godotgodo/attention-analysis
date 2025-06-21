import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import csv
import os

# List of names to process
names = ["yunus", "selin", "deniz", "eraycan"]
output_file = "raw_data.csv"

class AttentionAnalyzer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
            (0.0, -330.0, -65.0)    # Chin
        ], dtype="double")

        # Camera matrix
        self.focal_length = 1000
        self.center = (320, 240)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Initialize previous frame for movement detection
        self.prev_frame = None
        self.prev_landmarks = None
        self.prev_time = time.time()

    def calculate_eye_openness(self, shape, eye_indices):
        """Calculate eye openness ratio"""
        # Get eye landmarks
        eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_indices])
        
        # Calculate vertical distance (height)
        height = distance.euclidean(eye_points[1], eye_points[4])
        
        # Calculate horizontal distance (width)
        width = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate ratio
        ratio = height / width
        return ratio

    def calculate_eye_direction(self, shape, eye_indices):
        """Calculate eye direction (x, y) relative to eye center"""
        # Get eye landmarks
        eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_indices])
        
        # Calculate eye center (average of all eye landmarks)
        eye_center = np.mean(eye_points, axis=0)
        
        # Estimate iris position (using middle point of eye)
        iris_x = (eye_points[1][0] + eye_points[4][0]) / 2
        iris_y = (eye_points[1][1] + eye_points[4][1]) / 2
        
        # Calculate direction vector from center to iris
        direction_x = (iris_x - eye_center[0]) / distance.euclidean(eye_points[0], eye_points[3])
        direction_y = (iris_y - eye_center[1]) / distance.euclidean(eye_points[1], eye_points[4])
        
        return direction_x, direction_y

    def calculate_mouth_openness(self, shape):
        """Calculate mouth openness ratio"""
        # Get mouth landmarks
        mouth_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 68)])
        
        # Calculate vertical distance
        height = distance.euclidean(mouth_points[2], mouth_points[8])
        
        # Calculate horizontal distance
        width = distance.euclidean(mouth_points[0], mouth_points[6])
        
        # Calculate ratio
        ratio = height / width
        return ratio

    def calculate_face_movement(self, current_landmarks, prev_landmarks):
        """Calculate face movement between frames"""
        if prev_landmarks is None:
            return 0.0
        
        # Calculate average movement of all landmarks
        movement = np.mean(np.sqrt(np.sum((current_landmarks - prev_landmarks) ** 2, axis=1)))
        return movement

    def calculate_body_movement(self, current_frame, prev_frame):
        """Calculate body movement using optical flow"""
        if prev_frame is None:
            return 0.0
        
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude of movement
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def analyze_frame(self, frame):
        features = {
            'face_movement': 0.0,
            'body_movement': 0.0,
            'eye_openness_rate': 0.0,
            'eye_direction_x': 0.0,
            'eye_direction_y': 0.0,
            'mouth_openness_rate': 0.0,
            'yaw_angle': 0.0,
            'pitch_angle': 0.0,
            'roll_angle': 0.0,
            'isAttentive': False
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return features

        # Get the first face detected
        face = faces[0]
        shape = self.predictor(gray, face)
        
        # Convert landmarks to numpy array for movement calculation
        current_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # Calculate face movement
        features['face_movement'] = self.calculate_face_movement(current_landmarks, self.prev_landmarks)
        
        # Calculate body movement
        features['body_movement'] = self.calculate_body_movement(frame, self.prev_frame)
        
        # Calculate eye openness
        left_eye = self.calculate_eye_openness(shape, range(36, 42))
        right_eye = self.calculate_eye_openness(shape, range(42, 48))
        features['eye_openness_rate'] = (left_eye + right_eye) / 2
        
        # Calculate eye direction
        left_eye_dir_x, left_eye_dir_y = self.calculate_eye_direction(shape, range(36, 42))
        right_eye_dir_x, right_eye_dir_y = self.calculate_eye_direction(shape, range(42, 48))
        features['eye_direction_x'] = (left_eye_dir_x + right_eye_dir_x) / 2
        features['eye_direction_y'] = (left_eye_dir_y + right_eye_dir_y) / 2
        
        # Calculate mouth openness
        features['mouth_openness_rate'] = self.calculate_mouth_openness(shape)
        
        # Calculate head pose angles
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(36).x, shape.part(36).y),  # Left eye
            (shape.part(45).x, shape.part(45).y),  # Right eye
            (shape.part(48).x, shape.part(48).y),  # Left mouth corner
            (shape.part(54).x, shape.part(54).y),  # Right mouth corner
            (shape.part(8).x, shape.part(8).y)     # Chin
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, None
        )

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            features['yaw_angle'] = angles[0]
            features['pitch_angle'] = angles[1]
            features['roll_angle'] = angles[2]
        
        features['isAttentive'] = True

        # Update previous frame and landmarks
        self.prev_frame = frame.copy()
        self.prev_landmarks = current_landmarks

        return features

def process_video(name, analyzer, csv_writer, is_first_video):
    video_file = f"./clean_data/{name}_clean.mp4"
    
    # Check if the video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file not found for {name}: {video_file}")
        return
    
    print(f"Processing video for {name}: {video_file}")
    
    cap = cv2.VideoCapture(video_file)
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Reset analyzer's previous frame and landmarks
    analyzer.prev_frame = None
    analyzer.prev_landmarks = None
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate timestamp
        timestamp = frame_number / fps
        
        # Format timestamp as minutes:seconds:milliseconds
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp % 1) * 1000)
        formatted_timestamp = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
        
        # Analyze frame
        features = analyzer.analyze_frame(frame)
        
        # Write features to CSV
        row_data = {
            'frame_number': frame_number,
            'timestamp': formatted_timestamp,
            'id': name,
            **features
        }
        csv_writer.writerow(row_data)
        
        frame_number += 1
        
        # Print progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Processed frame {frame_number}/{frame_count} ({frame_number/frame_count*100:.1f}%) for {name}")
    
    print(f"Completed processing {frame_number} frames for {name}")
    cap.release()

def main():
    # Set to True if you want to see the video while processing
    display_video = False
    
    analyzer = AttentionAnalyzer()
    
    # Prepare CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['frame_number', 'timestamp', 'id', 'face_movement', 'body_movement', 
                    'eye_openness_rate', 'eye_direction_x', 'eye_direction_y', 
                    'mouth_openness_rate', 'yaw_angle', 'pitch_angle', 'roll_angle', 
                    'isAttentive']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each name sequentially
        for i, name in enumerate(names):
            is_first_video = (i == 0)
            process_video(name, analyzer, writer, is_first_video)
    
    print(f"Analysis complete. Results for all subjects saved to {output_file}")

if __name__ == "__main__":
    main()
