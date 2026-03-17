"""
Enhanced Violence Detector
Combines YOLO person detection with CLIP classification for better accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

# Import CLIP model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Model as CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP model not available")


class ViolenceDetector:
    """
    Enhanced violence detection using person detection + CLIP classification
    
    This detector:
    1. Uses YOLO to detect persons in the frame
    2. Uses CLIP to classify the overall scene for violence
    3. Combines both signals for more accurate detection
    """
    
    VIOLENCE_LABELS = {
        'fight on a street',
        'street violence',
        'physical fight',
        'people fighting',
        'violent altercation',
        'assault',
        'punching',
        'kicking',
        'violence in office',
        'fire in office',
        'fire on a street',
        'robbery',
        'attack'
    }
    
    ACCIDENT_LABELS = {
        'car crash',
        'car accident',
        'vehicle collision',
        'road accident',
        'traffic accident',
        'motorcycle accident',
        'pedestrian hit by car',
        'car hitting pedestrian',
        'vehicle crash',
        'cars colliding'
    }
    
    def __init__(
        self,
        clip_model: Optional['CLIPModel'] = None,
        violence_threshold: float = 0.3,
        min_persons: int = 1
    ):
        """
        Initialize violence detector
        
        Args:
            clip_model: Pre-initialized CLIP model (will create if None)
            violence_threshold: Minimum confidence for violence classification
            min_persons: Minimum persons in frame to consider violence
        """
        self.violence_threshold = violence_threshold
        self.min_persons = min_persons
        
        # Initialize CLIP model if not provided
        if clip_model is not None:
            self.clip_model = clip_model
        elif CLIP_AVAILABLE:
            self.clip_model = CLIPModel()
        else:
            self.clip_model = None
            print("Warning: Violence detection will not work without CLIP model")
        
        # Track recent detections for temporal smoothing
        self.detection_history = []
        self.history_size = 5
    
    def detect(
        self,
        frame: np.ndarray,
        person_detections: List[Dict] = None
    ) -> Dict:
        """
        Detect violence in frame
        
        Args:
            frame: BGR image
            person_detections: Optional list of person detections from YOLO
            
        Returns:
            Detection result with:
                - is_violence: Boolean indicating violence detected
                - confidence: Detection confidence
                - label: Detected label
                - person_count: Number of persons in frame
        """
        result = {
            'is_violence': False,
            'is_accident': False,
            'confidence': 0.0,
            'label': 'normal',
            'person_count': 0,
            'description': '',
            'detection_type': 'normal'
        }
        
        if self.clip_model is None:
            return result
        
        # Get person count
        person_count = len(person_detections) if person_detections else 0
        result['person_count'] = person_count
        
        # Run CLIP prediction
        try:
            prediction = self.clip_model.predict(frame)
            label = prediction.get('label', 'Unknown')
            confidence = prediction.get('confidence', 0.0)
            
            # Check if label indicates violence
            is_violence_label = label.lower() in [l.lower() for l in self.VIOLENCE_LABELS]
            
            # Check if label indicates accident
            is_accident_label = label.lower() in [l.lower() for l in self.ACCIDENT_LABELS]
            
            # Apply threshold for violence
            if is_violence_label and confidence >= self.violence_threshold:
                result['is_violence'] = True
                result['confidence'] = confidence
                result['label'] = label
                result['description'] = f"Violence detected: {label}"
                result['detection_type'] = 'violence'
            # Apply threshold for accident
            elif is_accident_label and confidence >= self.violence_threshold:
                result['is_accident'] = True
                result['confidence'] = confidence
                result['label'] = label
                result['description'] = f"Accident detected: {label}"
                result['detection_type'] = 'accident'
            else:
                result['label'] = label
                result['confidence'] = confidence
            
            # Add to history for temporal smoothing
            self._update_history(result['is_violence'])
            
            # Apply temporal smoothing - require multiple positive detections
            if self._get_smoothed_detection():
                result['is_violence'] = True
            
        except Exception as e:
            print(f"Violence detection error: {e}")
        
        return result
    
    def _update_history(self, is_violence: bool):
        """Update detection history"""
        self.detection_history.append(is_violence)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
    
    def _get_smoothed_detection(self) -> bool:
        """
        Get temporally smoothed detection result
        Requires majority of recent frames to show violence
        """
        if len(self.detection_history) < 3:
            return False
        
        violence_count = sum(self.detection_history)
        return violence_count >= len(self.detection_history) // 2 + 1
    
    def reset_history(self):
        """Reset detection history"""
        self.detection_history.clear()
    
    def get_severity(self, confidence: float) -> str:
        """
        Determine alert severity based on confidence
        
        Args:
            confidence: Detection confidence (0-1)
            
        Returns:
            Severity level string
        """
        if confidence >= 0.8:
            return 'critical'
        elif confidence >= 0.6:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'


class IntrusionDetector:
    """
    Zone-based intrusion detection
    Detects when persons enter restricted areas
    """
    
    def __init__(self, zones: List[Dict] = None):
        """
        Initialize intrusion detector
        
        Args:
            zones: List of zone configurations
                Each zone: {'name': str, 'polygon': [[x,y], ...], 'enabled': bool}
        """
        self.zones = zones or []
        self.intrusion_events = {}  # Track active intrusions per zone
    
    def add_zone(self, name: str, polygon: List[List[int]], enabled: bool = True):
        """Add a detection zone"""
        self.zones.append({
            'name': name,
            'polygon': np.array(polygon, dtype=np.int32),
            'enabled': enabled
        })
    
    def clear_zones(self):
        """Clear all zones"""
        self.zones.clear()
        self.intrusion_events.clear()
    
    def detect(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Detect intrusions into restricted zones
        
        Args:
            frame: Current video frame
            tracked_objects: Dictionary of tracked persons from ObjectTracker
            
        Returns:
            List of intrusion events
        """
        intrusions = []
        
        for zone in self.zones:
            if not zone.get('enabled', True):
                continue
            
            polygon = zone['polygon']
            if isinstance(polygon, list):
                polygon = np.array(polygon, dtype=np.int32)
            
            for person_id, info in tracked_objects.items():
                centroid = info['centroid']
                
                # Check if person is inside zone
                inside = cv2.pointPolygonTest(polygon, centroid, False) >= 0
                
                event_key = f"{zone['name']}_{person_id}"
                
                if inside:
                    # New intrusion
                    if event_key not in self.intrusion_events:
                        self.intrusion_events[event_key] = {
                            'start_time': info['time_tracked'],
                            'person_id': person_id,
                            'zone': zone['name']
                        }
                        
                        intrusions.append({
                            'type': 'intrusion',
                            'zone_name': zone['name'],
                            'person_id': person_id,
                            'centroid': centroid,
                            'bbox': info['bbox'],
                            'confidence': 1.0,
                            'severity': 'high',
                            'description': f"Person {person_id} entered restricted zone: {zone['name']}"
                        })
                else:
                    # Person left zone
                    if event_key in self.intrusion_events:
                        del self.intrusion_events[event_key]
        
        return intrusions
    
    def draw_zones(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw zone boundaries on frame"""
        frame_copy = frame.copy()
        
        for zone in self.zones:
            if not zone.get('enabled', True):
                continue
            
            polygon = zone['polygon']
            if isinstance(polygon, list):
                polygon = np.array(polygon, dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(frame_copy, [polygon], True, color, thickness)
            
            # Draw zone name
            if len(polygon) > 0:
                x, y = polygon[0]
                cv2.putText(
                    frame_copy,
                    zone['name'],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        return frame_copy


class LoiteringDetector:
    """
    Detect persons who stay in an area for too long
    """
    
    def __init__(
        self,
        time_threshold: float = 60.0,
        movement_threshold: float = 50.0
    ):
        """
        Initialize loitering detector
        
        Args:
            time_threshold: Seconds before loitering is flagged
            movement_threshold: Pixels - if person moves less than this, considered stationary
        """
        self.time_threshold = time_threshold
        self.movement_threshold = movement_threshold
        self.person_positions = {}  # {person_id: {'first_pos': (x,y), 'first_time': t}}
        self.alerted_persons = set()  # Track who we've already alerted for
    
    def detect(
        self,
        tracked_objects: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Detect loitering behavior
        
        Args:
            tracked_objects: Dictionary of tracked persons
            
        Returns:
            List of loitering events
        """
        loitering_events = []
        current_ids = set(tracked_objects.keys())
        
        # Clean up old entries
        old_ids = set(self.person_positions.keys()) - current_ids
        for old_id in old_ids:
            del self.person_positions[old_id]
            self.alerted_persons.discard(old_id)
        
        for person_id, info in tracked_objects.items():
            centroid = info['centroid']
            time_tracked = info['time_tracked']
            
            if person_id not in self.person_positions:
                # First time seeing this person
                self.person_positions[person_id] = {
                    'first_pos': centroid,
                    'first_time': time_tracked
                }
                continue
            
            first_pos = self.person_positions[person_id]['first_pos']
            
            # Calculate distance moved from initial position
            dx = centroid[0] - first_pos[0]
            dy = centroid[1] - first_pos[1]
            distance_moved = np.sqrt(dx**2 + dy**2)
            
            # Check if person has been relatively stationary
            if distance_moved < self.movement_threshold:
                # Check time threshold
                if time_tracked >= self.time_threshold:
                    if person_id not in self.alerted_persons:
                        self.alerted_persons.add(person_id)
                        
                        loitering_events.append({
                            'type': 'loitering',
                            'person_id': person_id,
                            'centroid': centroid,
                            'bbox': info['bbox'],
                            'duration': time_tracked,
                            'confidence': min(time_tracked / self.time_threshold, 1.0),
                            'severity': 'medium',
                            'description': f"Person {person_id} loitering for {int(time_tracked)}s"
                        })
            else:
                # Person moved significantly, reset tracking
                self.person_positions[person_id] = {
                    'first_pos': centroid,
                    'first_time': time_tracked
                }
                self.alerted_persons.discard(person_id)
        
        return loitering_events
    
    def reset(self):
        """Reset detector state"""
        self.person_positions.clear()
        self.alerted_persons.clear()


class RunningDetector:
    """
    Detect sudden running or fast movement
    """
    
    def __init__(
        self,
        speed_threshold: float = 15.0,
        sustained_frames: int = 3
    ):
        """
        Initialize running detector
        
        Args:
            speed_threshold: Pixels per frame threshold for "running"
            sustained_frames: Number of frames speed must be high
        """
        self.speed_threshold = speed_threshold
        self.sustained_frames = sustained_frames
        self.speed_history = {}  # {person_id: [speeds]}
        self.alerted_persons = set()
    
    def detect(
        self,
        tracked_objects: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Detect running behavior
        
        Args:
            tracked_objects: Dictionary of tracked persons
            
        Returns:
            List of running events
        """
        running_events = []
        current_ids = set(tracked_objects.keys())
        
        # Clean up old entries
        old_ids = set(self.speed_history.keys()) - current_ids
        for old_id in old_ids:
            del self.speed_history[old_id]
            self.alerted_persons.discard(old_id)
        
        for person_id, info in tracked_objects.items():
            speed = info['speed']
            
            # Update speed history
            if person_id not in self.speed_history:
                self.speed_history[person_id] = []
            
            self.speed_history[person_id].append(speed)
            
            # Keep only recent history
            if len(self.speed_history[person_id]) > self.sustained_frames * 2:
                self.speed_history[person_id] = self.speed_history[person_id][-self.sustained_frames * 2:]
            
            # Check if running
            recent_speeds = self.speed_history[person_id][-self.sustained_frames:]
            if len(recent_speeds) >= self.sustained_frames:
                avg_speed = sum(recent_speeds) / len(recent_speeds)
                
                if avg_speed > self.speed_threshold:
                    if person_id not in self.alerted_persons:
                        self.alerted_persons.add(person_id)
                        
                        running_events.append({
                            'type': 'running',
                            'person_id': person_id,
                            'centroid': info['centroid'],
                            'bbox': info['bbox'],
                            'speed': avg_speed,
                            'confidence': min(avg_speed / (self.speed_threshold * 2), 1.0),
                            'severity': 'medium',
                            'description': f"Person {person_id} running (speed: {avg_speed:.1f})"
                        })
                else:
                    # Speed dropped, reset alert
                    self.alerted_persons.discard(person_id)
        
        return running_events
    
    def reset(self):
        """Reset detector state"""
        self.speed_history.clear()
        self.alerted_persons.clear()
