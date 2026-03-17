"""
Crowd Anomaly Detection
Detects unusual crowd behavior including:
- Crowd density anomalies (too many people in an area)
- Sudden crowd gathering
- Sudden crowd dispersal
- Mass movement direction analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from collections import deque
from datetime import datetime


class CrowdAnomalyDetector:
    """
    Detects crowd-related anomalies using person detection data
    
    Features:
    1. Density monitoring - alerts when crowd exceeds threshold
    2. Gathering detection - sudden increase in person count
    3. Dispersal detection - sudden decrease (panic indicator)
    4. Flow analysis - unusual mass movement patterns
    """
    
    def __init__(
        self,
        density_threshold: int = 10,
        gathering_threshold: int = 5,
        dispersal_threshold: int = 5,
        time_window: int = 30,
        zones: List[Dict] = None
    ):
        """
        Initialize crowd anomaly detector
        
        Args:
            density_threshold: Max persons in zone before alert
            gathering_threshold: Increase in count that triggers alert
            dispersal_threshold: Decrease in count that triggers alert
            time_window: Number of frames to analyze for trends
            zones: Optional zones for per-zone density monitoring
        """
        self.density_threshold = density_threshold
        self.gathering_threshold = gathering_threshold
        self.dispersal_threshold = dispersal_threshold
        self.time_window = time_window
        self.zones = zones or []
        
        # History tracking
        self.count_history = deque(maxlen=time_window)
        self.position_history = deque(maxlen=time_window)
        self.flow_vectors = deque(maxlen=time_window // 2)
        
        # Alert cooldowns
        self.last_density_alert = None
        self.last_gathering_alert = None
        self.last_dispersal_alert = None
        self.cooldown_seconds = 30
        
    def detect(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        tracked_objects: Dict = None
    ) -> List[Dict]:
        """
        Analyze crowd behavior and detect anomalies
        
        Args:
            frame: Current video frame
            detections: List of YOLO detections
            tracked_objects: Optional tracked object data
            
        Returns:
            List of crowd anomaly events
        """
        events = []
        current_count = len(detections)
        current_time = datetime.now()
        
        # Extract person centroids
        centroids = []
        for det in detections:
            if det.get('class_name', 'person').lower() == 'person':
                x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                centroids.append((cx, cy))
        
        # Update history
        self.count_history.append(current_count)
        self.position_history.append(centroids)
        
        # 1. Density Check
        density_event = self._check_density(current_count, current_time)
        if density_event:
            events.append(density_event)
        
        # 2. Gathering Detection (sudden increase)
        gathering_event = self._check_gathering(current_time)
        if gathering_event:
            events.append(gathering_event)
        
        # 3. Dispersal Detection (sudden decrease - panic)
        dispersal_event = self._check_dispersal(current_time)
        if dispersal_event:
            events.append(dispersal_event)
        
        # 4. Flow Analysis (mass movement direction)
        flow_event = self._analyze_flow(frame, current_time)
        if flow_event:
            events.append(flow_event)
        
        return events
    
    def _check_density(self, current_count: int, current_time: datetime) -> Optional[Dict]:
        """Check if crowd density exceeds threshold"""
        if current_count >= self.density_threshold:
            if self._can_alert('density', current_time):
                self.last_density_alert = current_time
                return {
                    'type': 'crowd_density',
                    'severity': self._get_density_severity(current_count),
                    'confidence': min(1.0, current_count / self.density_threshold),
                    'description': f'High crowd density detected: {current_count} persons',
                    'person_count': current_count
                }
        return None
    
    def _check_gathering(self, current_time: datetime) -> Optional[Dict]:
        """Detect sudden crowd gathering"""
        if len(self.count_history) < self.time_window // 2:
            return None
        
        # Compare recent average to earlier average
        mid = len(self.count_history) // 2
        early_avg = np.mean(list(self.count_history)[:mid])
        recent_avg = np.mean(list(self.count_history)[mid:])
        
        increase = recent_avg - early_avg
        
        if increase >= self.gathering_threshold:
            if self._can_alert('gathering', current_time):
                self.last_gathering_alert = current_time
                return {
                    'type': 'crowd_gathering',
                    'severity': 'high',
                    'confidence': min(1.0, increase / (self.gathering_threshold * 2)),
                    'description': f'Unusual crowd gathering: +{increase:.0f} persons rapidly',
                    'increase': increase
                }
        return None
    
    def _check_dispersal(self, current_time: datetime) -> Optional[Dict]:
        """Detect sudden crowd dispersal (panic indicator)"""
        if len(self.count_history) < self.time_window // 2:
            return None
        
        mid = len(self.count_history) // 2
        early_avg = np.mean(list(self.count_history)[:mid])
        recent_avg = np.mean(list(self.count_history)[mid:])
        
        decrease = early_avg - recent_avg
        
        if decrease >= self.dispersal_threshold:
            if self._can_alert('dispersal', current_time):
                self.last_dispersal_alert = current_time
                return {
                    'type': 'crowd_dispersal',
                    'severity': 'critical',  # Dispersal can indicate panic/threat
                    'confidence': min(1.0, decrease / (self.dispersal_threshold * 2)),
                    'description': f'Sudden crowd dispersal detected: -{decrease:.0f} persons',
                    'decrease': decrease
                }
        return None
    
    def _analyze_flow(self, frame: np.ndarray, current_time: datetime) -> Optional[Dict]:
        """
        Analyze crowd flow direction using optical flow
        Detects unusual mass movement patterns
        """
        if len(self.position_history) < 3:
            return None
        
        # Get positions from recent frames
        prev_positions = list(self.position_history)[-3]
        curr_positions = list(self.position_history)[-1]
        
        if len(prev_positions) < 3 or len(curr_positions) < 3:
            return None
        
        # Calculate average movement direction
        movements = []
        for curr_pos in curr_positions:
            # Find closest previous position
            min_dist = float('inf')
            best_prev = None
            for prev_pos in prev_positions:
                dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                if dist < min_dist and dist < 100:  # Max matching distance
                    min_dist = dist
                    best_prev = prev_pos
            
            if best_prev:
                dx = curr_pos[0] - best_prev[0]
                dy = curr_pos[1] - best_prev[1]
                if abs(dx) > 5 or abs(dy) > 5:  # Minimum movement
                    movements.append((dx, dy))
        
        if len(movements) < 3:
            return None
        
        # Check for uniform movement direction (stampede/panic)
        angles = [np.arctan2(m[1], m[0]) for m in movements]
        angle_std = np.std(angles)
        avg_speed = np.mean([np.sqrt(m[0]**2 + m[1]**2) for m in movements])
        
        # Low angle variance + high speed = potential stampede
        if angle_std < 0.5 and avg_speed > 20:
            return {
                'type': 'crowd_stampede',
                'severity': 'critical',
                'confidence': min(1.0, avg_speed / 40),
                'description': f'Coordinated crowd movement detected (potential stampede)',
                'speed': avg_speed,
                'direction_variance': angle_std
            }
        
        return None
    
    def _can_alert(self, alert_type: str, current_time: datetime) -> bool:
        """Check if alert can be sent (cooldown check)"""
        last_alert_map = {
            'density': self.last_density_alert,
            'gathering': self.last_gathering_alert,
            'dispersal': self.last_dispersal_alert
        }
        
        last_alert = last_alert_map.get(alert_type)
        if last_alert is None:
            return True
        
        elapsed = (current_time - last_alert).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    def _get_density_severity(self, count: int) -> str:
        """Determine severity based on density level"""
        ratio = count / self.density_threshold
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        else:
            return 'medium'
    
    def get_crowd_stats(self) -> Dict:
        """Get current crowd statistics"""
        if not self.count_history:
            return {'current': 0, 'average': 0, 'max': 0, 'trend': 'stable'}
        
        counts = list(self.count_history)
        current = counts[-1] if counts else 0
        average = np.mean(counts)
        max_count = max(counts)
        
        # Determine trend
        if len(counts) >= 10:
            recent = np.mean(counts[-5:])
            earlier = np.mean(counts[:5])
            if recent > earlier + 2:
                trend = 'increasing'
            elif recent < earlier - 2:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'current': current,
            'average': round(average, 1),
            'max': max_count,
            'trend': trend
        }
    
    def draw_crowd_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw crowd statistics on frame"""
        stats = self.get_crowd_stats()
        h, w = frame.shape[:2]
        
        # Draw info box in top-right corner
        info_text = [
            f"Crowd: {stats['current']} persons",
            f"Avg: {stats['average']} | Max: {stats['max']}",
            f"Trend: {stats['trend']}"
        ]
        
        x_start = w - 200
        y_start = 100
        
        for i, text in enumerate(info_text):
            cv2.putText(
                frame, text, (x_start, y_start + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return frame
    
    def reset(self):
        """Reset detector state"""
        self.count_history.clear()
        self.position_history.clear()
        self.flow_vectors.clear()
        self.last_density_alert = None
        self.last_gathering_alert = None
        self.last_dispersal_alert = None
