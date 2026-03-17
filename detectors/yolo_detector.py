"""
YOLOv8 Person and Object Detector
Uses Ultralytics YOLOv8 for real-time person detection with bounding boxes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class YOLODetector:
    """
    YOLOv8-based person and object detector
    
    Attributes:
        model: YOLOv8 model instance
        confidence_threshold: Minimum confidence for detections
        classes: List of class IDs to detect (0 = person in COCO)
    """
    
    # COCO class names for reference
    COCO_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        39: 'bottle',
        41: 'cup',
        43: 'knife',
        67: 'cell phone',
        73: 'laptop'
    }
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
        device: str = 'auto'
    ):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model weights (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence score for detections
            classes: List of class IDs to detect. Default [0] for person only
            device: Device to run inference ('auto', 'cpu', 'cuda', '0', '1', etc.)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes if classes is not None else [0]  # Default: person only
        self.device = device
        
        # Load model
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device != 'auto':
            self.model.to(device)
        
        print(f"YOLOv8 detector initialized. Detecting classes: {[self.COCO_CLASSES.get(c, c) for c in self.classes]}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_id: Class ID from COCO dataset
                - class_name: Human-readable class name
                - center: (cx, cy) center point of bounding box
        """
        detections = []
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=self.classes,
            verbose=False
        )
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
                
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.COCO_CLASSES.get(class_id, f'class_{class_id}')
                
                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': (cx, cy),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                })
        
        return detections
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Convenience method to detect only persons
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of person detections
        """
        # Temporarily set classes to person only
        original_classes = self.classes
        self.classes = [0]
        
        detections = self.detect(frame)
        
        # Restore original classes
        self.classes = original_classes
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_label: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: BGR image to draw on
            detections: List of detection dictionaries
            color: BGR color for boxes
            thickness: Line thickness
            show_label: Whether to show class label
            show_confidence: Whether to show confidence score
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_label or show_confidence:
                label_parts = []
                if show_label:
                    label_parts.append(det['class_name'])
                if show_confidence:
                    label_parts.append(f"{det['confidence']:.2f}")
                
                label = ' '.join(label_parts)
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    frame_copy,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame_copy,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Draw center point
            cv2.circle(frame_copy, det['center'], 4, color, -1)
        
        return frame_copy
    
    def get_detection_count(self, detections: List[Dict], class_name: str = 'person') -> int:
        """
        Count detections of a specific class
        
        Args:
            detections: List of detection dictionaries
            class_name: Class name to count
            
        Returns:
            Number of detections matching the class
        """
        return sum(1 for d in detections if d['class_name'] == class_name)


# Test function
if __name__ == '__main__':
    import cv2
    
    # Initialize detector
    detector = YOLODetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        classes=[0]  # Person only
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        
        # Draw
        frame = detector.draw_detections(frame, detections)
        
        # Show person count
        count = detector.get_detection_count(detections, 'person')
        cv2.putText(
            frame,
            f'Persons: {count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('YOLOv8 Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
