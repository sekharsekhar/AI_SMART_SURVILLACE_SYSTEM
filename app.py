"""
AI-Based Intelligent Video Surveillance System
Enhanced Flask application with multi-activity detection and email alerts
"""

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
from model import Model
import time
import os
import yaml
from datetime import datetime
from werkzeug.utils import secure_filename

# Import new detection modules
try:
    from detectors.yolo_detector import YOLODetector, YOLO_AVAILABLE
except ImportError:
    YOLO_AVAILABLE = False
    YOLODetector = None

try:
    from detectors.violence_detector import ViolenceDetector, IntrusionDetector, LoiteringDetector, RunningDetector
    from detectors.crowd_detector import CrowdAnomalyDetector
except ImportError:
    ViolenceDetector = None
    CrowdAnomalyDetector = None

try:
    from tracking.tracker import ObjectTracker
except ImportError:
    ObjectTracker = None

try:
    from alerts.email_alert import EmailAlertEngine
except ImportError:
    EmailAlertEngine = None

try:
    from database.db import Database
except ImportError:
    Database = None

app = Flask(__name__)

# ==================== Configuration ====================

def load_config():
    """Load configuration from settings.yaml"""
    with open('settings.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Folder setup
UPLOAD_FOLDER = 'uploaded_videos'
SNAPSHOTS_FOLDER = 'snapshots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOTS_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# ==================== Initialize Models ====================

print("="*50)
print("AI-Based Intelligent Video Surveillance System")
print("="*50)
print("Initializing AI models...")

# CLIP model for violence classification
clip_model = Model()

# YOLOv8 for person detection
yolo_detector = None
if YOLO_AVAILABLE and YOLODetector:
    try:
        yolo_settings = config.get('yolo-settings', {})
        yolo_detector = YOLODetector(
            model_path=yolo_settings.get('model', 'yolov8n.pt'),
            confidence_threshold=yolo_settings.get('confidence-threshold', 0.5),
            classes=yolo_settings.get('classes', [0])
        )
        print("✓ YOLOv8 detector initialized")
    except Exception as e:
        print(f"✗ YOLO detector not available: {e}")
else:
    print("✗ YOLO detector not available (install ultralytics)")

# Object tracker
tracker = None
if ObjectTracker:
    tracking_settings = config.get('tracking-settings', {})
    tracker = ObjectTracker(
        max_disappeared=tracking_settings.get('max-disappeared', 30),
        max_distance=tracking_settings.get('max-distance', 100),
        trajectory_length=tracking_settings.get('trajectory-length', 50)
    )
    print("✓ Object tracker initialized")

# Activity detectors
detection_settings = config.get('detection-settings', {})

violence_detector = None
if ViolenceDetector:
    violence_detector = ViolenceDetector(
        clip_model=clip_model,
        violence_threshold=detection_settings.get('violence', {}).get('threshold', 0.3)
    )
    print("✓ Violence detector initialized")

intrusion_detector = None
if 'IntrusionDetector' in dir():
    from detectors.violence_detector import IntrusionDetector
    intrusion_detector = IntrusionDetector(zones=config.get('zones', []))
    print("✓ Intrusion detector initialized")

loitering_detector = None
if 'LoiteringDetector' in dir():
    from detectors.violence_detector import LoiteringDetector
    loitering_settings = detection_settings.get('loitering', {})
    loitering_detector = LoiteringDetector(
        time_threshold=loitering_settings.get('time-threshold', 60),
        movement_threshold=loitering_settings.get('movement-threshold', 50)
    )
    print("✓ Loitering detector initialized")

running_detector = None
if 'RunningDetector' in dir():
    from detectors.violence_detector import RunningDetector
    running_settings = detection_settings.get('running', {})
    running_detector = RunningDetector(
        speed_threshold=running_settings.get('speed-threshold', 15.0),
        sustained_frames=running_settings.get('sustained-frames', 3)
    )
    print("✓ Running detector initialized")

crowd_detector = None
if CrowdAnomalyDetector:
    crowd_settings = detection_settings.get('crowd', {})
    crowd_detector = CrowdAnomalyDetector(
        density_threshold=crowd_settings.get('density-threshold', 10),
        gathering_threshold=crowd_settings.get('gathering-threshold', 5),
        dispersal_threshold=crowd_settings.get('dispersal-threshold', 5),
        time_window=crowd_settings.get('time-window', 30)
    )
    print("✓ Crowd anomaly detector initialized")

# Email alert engine
alert_engine = None
if EmailAlertEngine:
    alert_config = config.get('alerts', {}).get('smtp', {})
    alert_engine = EmailAlertEngine(
        smtp_host=alert_config.get('host', ''),
        smtp_port=alert_config.get('port', 587),
        smtp_username=alert_config.get('username', ''),
        smtp_password=alert_config.get('password', ''),
        from_address=alert_config.get('from_address', ''),
        recipients=alert_config.get('recipients', []),
        use_tls=alert_config.get('use_tls', True),
        cooldown_seconds=config.get('alerts', {}).get('cooldown_seconds', 30),
        enabled=alert_config.get('enabled', False)
    )
    print(f"✓ Email alerts: {'Configured' if alert_engine.is_configured() else 'Not configured'}")

# Database
database = None
if Database:
    db_config = config.get('database', {})
    database = Database(db_path=db_config.get('path', 'data/surveillance.db'))
    print("✓ Database initialized")

print("="*50)

# ==================== Processing Settings ====================

FRAME_SKIP = 3
TARGET_WIDTH = 640
TARGET_FPS = 12

# Violence detection labels (for legacy CLIP-only detection)
VIOLENCE_LABELS = {'fight on a street', 'street violence', 'violence in office', 'fire in office', 'fire on a street'}

# ==================== Helper Functions ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_uploaded_videos():
    """Get list of uploaded video files"""
    videos = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if allowed_file(filename):
                videos.append(filename)
    return sorted(videos)

def send_alert_if_needed(activity_type, severity, confidence, frame, description=''):
    """Send email alert and log to database"""
    print(f"[DEBUG] send_alert_if_needed called: {activity_type}, severity={severity}, conf={confidence}")
    snapshot_path = None
    
    # Save snapshot
    if frame is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path = os.path.join(SNAPSHOTS_FOLDER, f'{activity_type}_{timestamp}.jpg')
        try:
            cv2.imwrite(snapshot_path, frame)
            print(f"[DEBUG] Snapshot saved: {snapshot_path}")
        except Exception as e:
            print(f"[DEBUG] Snapshot save failed: {e}")
            snapshot_path = None
    
    # Log to database
    if database:
        event = database.log_event(
            activity_type=activity_type,
            severity=severity,
            confidence=confidence,
            description=description,
            snapshot_path=snapshot_path
        )
        print(f"[DEBUG] Event logged to database: ID={event.id if event else 'None'}")
        
        # Send email alert
        if alert_engine:
            print(f"[DEBUG] Alert engine exists, enabled={alert_engine.enabled}, configured={alert_engine.is_configured()}")
            print(f"[DEBUG] SMTP: host={alert_engine.smtp_host}, user={alert_engine.smtp_username}, recipients={alert_engine.recipients}")
            if alert_engine.enabled and alert_engine.is_configured():
                sent = alert_engine.send_alert(
                    activity_type=activity_type,
                    priority=severity,
                    confidence=confidence,
                    description=description,
                    frame=frame,
                    snapshot_path=snapshot_path
                )
                print(f"[DEBUG] Email send result: {sent}")
                if sent:
                    database.mark_email_sent(event.id)
            else:
                print(f"[DEBUG] Alert engine not sending - enabled:{alert_engine.enabled}, configured:{alert_engine.is_configured()}")
        else:
            print("[DEBUG] No alert engine available")

def draw_status_panel(frame, violence_result, person_count, alert_count):
    """Draw detection status panel on frame"""
    h, w = frame.shape[:2]
    
    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Determine status
    is_danger = violence_result.get('is_violence', False) or violence_result.get('is_accident', False) or alert_count > 0
    status_color = (0, 0, 255) if is_danger else (0, 255, 0)
    
    if violence_result.get('is_violence'):
        status_text = f"[!] VIOLENCE DETECTED: {violence_result.get('label', 'Unknown')}"
    elif violence_result.get('is_accident'):
        status_text = f"[!] ACCIDENT DETECTED: {violence_result.get('label', 'Unknown')}"
    elif alert_count > 0:
        status_text = f"[!] ALERT: {alert_count} abnormal activity detected"
    else:
        status_text = "[OK] SAFE - No Threats Detected"
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Stats bar
    confidence = violence_result.get('confidence', 0)
    stats_text = f"Persons: {person_count} | Confidence: {confidence:.2f}"
    cv2.putText(frame, stats_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, timestamp, (w - 100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

# ==================== Video Processing ====================

def generate_frames(video_path):
    """
    Enhanced frame generator with multi-activity detection
    Supports both video files and webcam
    """
    # Reset detector states for new video
    if tracker:
        tracker.reset()
    if violence_detector:
        violence_detector.reset_history()
    if loitering_detector:
        loitering_detector.reset()
    if running_detector:
        running_detector.reset()
    if crowd_detector:
        crowd_detector.reset()
    
    # Open video source
    if video_path == 'webcam' or video_path == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {video_path}")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Resize frame
        original_height, original_width = frame.shape[:2]
        aspect_ratio = original_width / original_height
        target_height = int(TARGET_WIDTH / aspect_ratio)
        frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
        
        # Convert to RGB for models
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection results
        detections = []
        tracked_objects = {}
        alerts = []
        violence_result = {'is_violence': False, 'is_accident': False, 'confidence': 0, 'label': 'normal'}
        
        # 1. YOLO Person Detection
        if yolo_detector:
            detections = yolo_detector.detect(frame)
            frame = yolo_detector.draw_detections(frame, detections, color=(0, 255, 0))
        
        # 2. Object Tracking
        if tracker and len(detections) > 0:
            tracked_objects = tracker.update(detections)
            frame = tracker.draw_trajectories(frame, color=(255, 255, 0), thickness=1)
        elif tracker:
            tracked_objects = tracker.update([])
        
        # 3. Violence & Accident Detection
        if violence_detector:
            violence_result = violence_detector.detect(frame_rgb, detections)
            if violence_result.get('is_violence'):
                alerts.append({
                    'type': 'violence',
                    'severity': violence_detector.get_severity(violence_result['confidence']),
                    'confidence': violence_result['confidence'],
                    'description': violence_result.get('description', '')
                })
            if violence_result.get('is_accident'):
                alerts.append({
                    'type': 'accident',
                    'severity': 'critical',  # Accidents are always critical
                    'confidence': violence_result['confidence'],
                    'description': violence_result.get('description', '')
                })
        else:
            # Fallback to basic CLIP detection
            prediction = clip_model.predict(image=frame_rgb)
            label = prediction.get('label', 'Unknown')
            if label in VIOLENCE_LABELS:
                violence_result = {'is_violence': True, 'confidence': 0.5, 'label': label}
                alerts.append({
                    'type': 'violence',
                    'severity': 'high',
                    'confidence': 0.5,
                    'description': f'Violence detected: {label}'
                })
        
        # 4. Loitering Detection
        if loitering_detector and detection_settings.get('loitering', {}).get('enabled', True):
            loitering_events = loitering_detector.detect(tracked_objects)
            alerts.extend(loitering_events)
        
        # 5. Running Detection
        if running_detector and detection_settings.get('running', {}).get('enabled', True):
            running_events = running_detector.detect(tracked_objects)
            alerts.extend(running_events)
        
        # 6. Intrusion Detection
        if intrusion_detector and detection_settings.get('intrusion', {}).get('enabled', True):
            if len(intrusion_detector.zones) > 0:
                intrusion_events = intrusion_detector.detect(frame, tracked_objects)
                alerts.extend(intrusion_events)
                frame = intrusion_detector.draw_zones(frame)
        
        # 7. Crowd Anomaly Detection
        if crowd_detector and detection_settings.get('crowd', {}).get('enabled', True):
            crowd_events = crowd_detector.detect(frame, detections, tracked_objects)
            alerts.extend(crowd_events)
            frame = crowd_detector.draw_crowd_info(frame)
        
        # Process alerts
        for alert in alerts:
            send_alert_if_needed(
                activity_type=alert.get('type', 'unknown'),
                severity=alert.get('severity', 'medium'),
                confidence=alert.get('confidence', 1.0),
                frame=frame,
                description=alert.get('description', '')
            )
        
        # Draw status panel
        person_count = len(detections) if detections else 0
        frame = draw_status_panel(frame, violence_result, person_count, len(alerts))
        
        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Frame rate control
        elapsed = time.time() - start_time
        target_time = 1.0 / TARGET_FPS
        if elapsed < target_time:
            time.sleep(target_time - elapsed)
        start_time = time.time()
    
    cap.release()

# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard"""
    uploaded_videos = get_uploaded_videos()
    stats = database.get_statistics(days=7) if database else {}
    alert_configured = alert_engine.is_configured() if alert_engine else False
    
    return render_template('index.html',
                           uploaded_videos=uploaded_videos,
                           stats=stats,
                           alert_configured=alert_configured,
                           yolo_enabled=yolo_detector is not None)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video_file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['video_file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('single_uploaded_video', filename=filename))
    
    return "File upload failed or file type not allowed.", 400

@app.route('/single_video/<int:index>')
def single_video(index):
    """View uploaded video by index"""
    videos = get_uploaded_videos()
    if 0 <= index < len(videos):
        return redirect(url_for('single_uploaded_video', filename=videos[index]))
    return "Video not found", 404

@app.route('/single_uploaded_video/<filename>')
def single_uploaded_video(filename):
    """Video analysis page"""
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(full_path) or not allowed_file(filename):
        return "Video not found", 404
    
    return render_template('single_video.html',
                           video_filename=filename,
                           feed_type='uploaded',
                           identifier=filename)

@app.route('/webcam')
def webcam_view():
    """Live webcam detection page"""
    return render_template('single_video.html',
                           video_filename='Live Webcam',
                           feed_type='webcam',
                           identifier='0')

@app.route('/video_feed/<feed_type>/<identifier>')
def video_feed(feed_type, identifier):
    """Video stream endpoint with detection"""
    try:
        if feed_type == 'webcam':
            return Response(generate_frames('webcam'),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        elif feed_type == 'uploaded':
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], identifier)
            if os.path.exists(video_path):
                return Response(generate_frames(video_path),
                               mimetype='multipart/x-mixed-replace; boundary=frame')
        
        return "Video not found", 404
    except Exception as e:
        return str(e), 500

# ==================== API Endpoints ====================

@app.route('/api/events')
def api_get_events():
    """Get detection events"""
    if not database:
        return jsonify({'events': [], 'count': 0})
    
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    activity_type = request.args.get('type', None)
    
    events = database.get_events(limit=limit, offset=offset, activity_type=activity_type)
    return jsonify({'events': events, 'count': len(events)})

@app.route('/api/stats')
def api_get_stats():
    """Get detection statistics"""
    if not database:
        return jsonify({})
    
    days = request.args.get('days', 7, type=int)
    stats = database.get_statistics(days=days)
    return jsonify(stats)

@app.route('/api/test_email', methods=['POST'])
def api_test_email():
    """Test email configuration"""
    if not alert_engine or not alert_engine.is_configured():
        return jsonify({'success': False, 'message': 'Email not configured'})
    
    result = alert_engine.send_test_email()
    return jsonify(result)

@app.route('/api/config/smtp', methods=['POST'])
def api_update_smtp():
    """Update SMTP configuration"""
    if not alert_engine:
        return jsonify({'success': False, 'message': 'Alert engine not available'})
    
    data = request.json
    
    alert_engine.smtp_host = data.get('host', '')
    alert_engine.smtp_port = data.get('port', 587)
    alert_engine.smtp_username = data.get('username', '')
    alert_engine.smtp_password = data.get('password', '')
    alert_engine.from_address = data.get('from_address', '')
    alert_engine.recipients = data.get('recipients', [])
    alert_engine.enabled = data.get('enabled', False)
    
    result = alert_engine.test_connection()
    return jsonify(result)

# ==================== Main ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)