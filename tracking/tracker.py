"""
Object Tracker using Simple Centroid Tracking
Lightweight tracking without deep-sort dependency for academic projects
Tracks persons across frames and maintains trajectory history
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, deque
from scipy.spatial import distance as dist
import time


class ObjectTracker:
    """
    Simple centroid-based object tracker with trajectory history
    
    This tracker assigns unique IDs to detected objects and tracks them
    across frames based on centroid distance matching.
    
    Attributes:
        max_disappeared: Maximum frames an object can disappear before removal
        max_distance: Maximum distance threshold for matching
        trajectory_length: Number of positions to keep in history
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: int = 100,
        trajectory_length: int = 50
    ):
        """
        Initialize the tracker
        
        Args:
            max_disappeared: Frames before an object is deregistered
            max_distance: Maximum centroid distance for matching
            trajectory_length: Number of historical positions to store
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {id: centroid}
        self.disappeared = OrderedDict()  # {id: disappeared_count}
        self.trajectories = OrderedDict()  # {id: deque of positions}
        self.bboxes = OrderedDict()  # {id: current bbox}
        self.timestamps = OrderedDict()  # {id: first_seen_time}
        self.velocities = OrderedDict()  # {id: (vx, vy)}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectory_length = trajectory_length
    
    def register(self, centroid: Tuple[int, int], bbox: List[int]) -> int:
        """
        Register a new object with the next available ID
        
        Args:
            centroid: (x, y) center point
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Assigned object ID
        """
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.trajectories[object_id] = deque(maxlen=self.trajectory_length)
        self.trajectories[object_id].append(centroid)
        self.bboxes[object_id] = bbox
        self.timestamps[object_id] = time.time()
        self.velocities[object_id] = (0, 0)
        
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id: int):
        """
        Remove an object from tracking
        
        Args:
            object_id: ID of object to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trajectories[object_id]
        del self.bboxes[object_id]
        del self.timestamps[object_id]
        del self.velocities[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'center' and 'bbox' keys
            
        Returns:
            Dictionary mapping object IDs to tracking info:
                - centroid: Current position
                - bbox: Current bounding box
                - trajectory: Position history
                - velocity: (vx, vy) pixels per frame
                - time_tracked: Seconds since first seen
        """
        # Extract centroids and bboxes from detections
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self._get_tracking_info()
        
        input_centroids = np.array([d['center'] for d in detections])
        input_bboxes = [d['bbox'] for d in detections]
        
        # If no objects being tracked, register all detections
        if len(self.objects) == 0:
            for i, (centroid, bbox) in enumerate(zip(input_centroids, input_bboxes)):
                self.register(tuple(centroid), bbox)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                new_centroid = tuple(input_centroids[col])
                
                # Update object
                self.objects[object_id] = new_centroid
                self.trajectories[object_id].append(new_centroid)
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                # Calculate velocity
                vx = new_centroid[0] - old_centroid[0]
                vy = new_centroid[1] - old_centroid[1]
                self.velocities[object_id] = (vx, vy)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing objects (disappeared)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Handle unmatched new detections (new objects)
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), input_bboxes[col])
        
        return self._get_tracking_info()
    
    def _get_tracking_info(self) -> Dict[int, Dict]:
        """
        Get current tracking information for all objects
        
        Returns:
            Dictionary with tracking info per object ID
        """
        current_time = time.time()
        result = {}
        
        for object_id in self.objects:
            result[object_id] = {
                'centroid': self.objects[object_id],
                'bbox': self.bboxes[object_id],
                'trajectory': list(self.trajectories[object_id]),
                'velocity': self.velocities[object_id],
                'speed': np.sqrt(self.velocities[object_id][0]**2 + self.velocities[object_id][1]**2),
                'time_tracked': current_time - self.timestamps[object_id],
                'disappeared': self.disappeared[object_id]
            }
        
        return result
    
    def get_trajectory(self, object_id: int) -> List[Tuple[int, int]]:
        """
        Get trajectory history for an object
        
        Args:
            object_id: Object ID
            
        Returns:
            List of (x, y) positions
        """
        if object_id in self.trajectories:
            return list(self.trajectories[object_id])
        return []
    
    def get_velocity(self, object_id: int) -> Tuple[float, float]:
        """
        Get current velocity of an object
        
        Args:
            object_id: Object ID
            
        Returns:
            (vx, vy) velocity in pixels per frame
        """
        if object_id in self.velocities:
            return self.velocities[object_id]
        return (0, 0)
    
    def get_speed(self, object_id: int) -> float:
        """
        Get speed magnitude of an object
        
        Args:
            object_id: Object ID
            
        Returns:
            Speed in pixels per frame
        """
        vx, vy = self.get_velocity(object_id)
        return np.sqrt(vx**2 + vy**2)
    
    def get_time_stationary(self, object_id: int, threshold: float = 5.0) -> float:
        """
        Calculate how long an object has been stationary
        
        Args:
            object_id: Object ID
            threshold: Speed threshold below which object is considered stationary
            
        Returns:
            Seconds the object has been stationary (approximate)
        """
        if object_id not in self.trajectories:
            return 0.0
        
        trajectory = list(self.trajectories[object_id])
        if len(trajectory) < 2:
            return 0.0
        
        # Count frames where movement was below threshold
        stationary_frames = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            
            if speed < threshold:
                stationary_frames += 1
            else:
                stationary_frames = 0  # Reset if moved
        
        # Estimate time (assuming ~10 FPS)
        return stationary_frames / 10.0
    
    def draw_trajectories(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw trajectory lines on frame
        
        Args:
            frame: Image to draw on
            color: BGR color for lines
            thickness: Line thickness
            
        Returns:
            Frame with trajectories drawn
        """
        import cv2
        frame_copy = frame.copy()
        
        for object_id, trajectory in self.trajectories.items():
            points = list(trajectory)
            
            # Draw trajectory line
            for i in range(1, len(points)):
                cv2.line(frame_copy, points[i-1], points[i], color, thickness)
            
            # Draw current position with ID
            if points:
                current = points[-1]
                cv2.circle(frame_copy, current, 5, color, -1)
                cv2.putText(
                    frame_copy,
                    f'ID:{object_id}',
                    (current[0] + 10, current[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        return frame_copy
    
    def reset(self):
        """Reset all tracking state"""
        self.next_object_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.trajectories.clear()
        self.bboxes.clear()
        self.timestamps.clear()
        self.velocities.clear()


# Test function
if __name__ == '__main__':
    import cv2
    
    # Simulated detections for testing
    tracker = ObjectTracker(max_disappeared=10, max_distance=50)
    
    # Simulate some detections
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections = [
        {'center': (100, 100), 'bbox': [80, 80, 120, 120]},
        {'center': (300, 200), 'bbox': [280, 180, 320, 220]},
    ]
    
    tracking_info = tracker.update(detections)
    print("Frame 1 tracking:", tracking_info)
    
    # Move objects
    detections = [
        {'center': (110, 105), 'bbox': [90, 85, 130, 125]},
        {'center': (310, 210), 'bbox': [290, 190, 330, 230]},
    ]
    
    tracking_info = tracker.update(detections)
    print("Frame 2 tracking:", tracking_info)
    
    # Add new object
    detections = [
        {'center': (120, 110), 'bbox': [100, 90, 140, 130]},
        {'center': (320, 220), 'bbox': [300, 200, 340, 240]},
        {'center': (500, 300), 'bbox': [480, 280, 520, 320]},
    ]
    
    tracking_info = tracker.update(detections)
    print("Frame 3 tracking:", tracking_info)
