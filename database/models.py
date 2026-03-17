"""
SQLAlchemy Database Models
Defines the schema for storing detection events
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class Event(Base):
    """
    Detection event record
    
    Stores information about detected abnormal activities
    """
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Source information
    camera_id = Column(String(50), default='default')
    camera_name = Column(String(100), default='Camera 1')
    
    # Detection details
    activity_type = Column(String(50), nullable=False)  # violence, intrusion, loitering, running
    severity = Column(String(20), default='medium')  # critical, high, medium, low
    confidence = Column(Float, default=0.0)
    
    # Location
    zone_name = Column(String(100), default='Main Area')
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    
    # Additional info
    description = Column(Text, nullable=True)
    snapshot_path = Column(String(255), nullable=True)
    
    # Notification status
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime, nullable=True)
    
    # Object tracking
    person_id = Column(Integer, nullable=True)  # Tracked person ID
    
    def __repr__(self):
        return f"<Event(id={self.id}, type={self.activity_type}, severity={self.severity}, time={self.timestamp})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'activity_type': self.activity_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'zone_name': self.zone_name,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2] if self.bbox_x1 else None,
            'description': self.description,
            'snapshot_path': self.snapshot_path,
            'email_sent': self.email_sent,
            'person_id': self.person_id
        }


class Camera(Base):
    """
    Camera/video source configuration
    """
    __tablename__ = 'cameras'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    source = Column(String(255), nullable=False)  # file path, webcam index, or URL
    source_type = Column(String(20), default='file')  # file, webcam, rtsp
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name={self.name}, type={self.source_type})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'source': self.source,
            'source_type': self.source_type,
            'enabled': self.enabled
        }


class AlertSettings(Base):
    """
    Alert configuration settings (stored in DB for easy management)
    """
    __tablename__ = 'alert_settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AlertSettings(key={self.key}, value={self.value})>"


class DetectionStats(Base):
    """
    Aggregated detection statistics per day
    """
    __tablename__ = 'detection_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    activity_type = Column(String(50), nullable=False)
    count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<DetectionStats(date={self.date}, type={self.activity_type}, count={self.count})>"


# Create engine and tables
def init_db(db_path: str = 'data/surveillance.db'):
    """Initialize database and create tables"""
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Create a new database session"""
    Session = sessionmaker(bind=engine)
    return Session()
