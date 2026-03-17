"""
Database Operations
Provides high-level database operations for the surveillance system
"""

import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, Event, Camera, AlertSettings, DetectionStats


class Database:
    """
    Database manager for the surveillance system
    
    Handles all database operations including:
    - Event logging
    - Statistics queries
    - Configuration storage
    """
    
    def __init__(self, db_path: str = 'data/surveillance.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionFactory()
    
    # ==================== Event Operations ====================
    
    def log_event(
        self,
        activity_type: str,
        severity: str = 'medium',
        confidence: float = 0.0,
        camera_id: str = 'default',
        camera_name: str = 'Camera 1',
        zone_name: str = 'Main Area',
        description: str = None,
        snapshot_path: str = None,
        bbox: List[int] = None,
        person_id: int = None
    ) -> Event:
        """
        Log a detection event to the database
        
        Args:
            activity_type: Type of activity detected
            severity: Alert severity level
            confidence: Detection confidence (0-1)
            camera_id: Camera identifier
            camera_name: Camera display name
            zone_name: Detection zone name
            description: Event description
            snapshot_path: Path to saved snapshot
            bbox: Bounding box [x1, y1, x2, y2]
            person_id: Tracked person ID
            
        Returns:
            Created Event object
        """
        session = self.get_session()
        
        try:
            event = Event(
                activity_type=activity_type,
                severity=severity,
                confidence=confidence,
                camera_id=camera_id,
                camera_name=camera_name,
                zone_name=zone_name,
                description=description,
                snapshot_path=snapshot_path,
                person_id=person_id
            )
            
            if bbox and len(bbox) == 4:
                event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2 = bbox
            
            session.add(event)
            session.commit()
            
            # Refresh to get auto-generated fields
            session.refresh(event)
            
            return event
            
        finally:
            session.close()
    
    def get_events(
        self,
        limit: int = 50,
        offset: int = 0,
        activity_type: str = None,
        severity: str = None,
        camera_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict]:
        """
        Get events with optional filtering
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            activity_type: Filter by activity type
            severity: Filter by severity
            camera_id: Filter by camera
            start_date: Filter events after this date
            end_date: Filter events before this date
            
        Returns:
            List of event dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(Event).order_by(desc(Event.timestamp))
            
            if activity_type:
                query = query.filter(Event.activity_type == activity_type)
            if severity:
                query = query.filter(Event.severity == severity)
            if camera_id:
                query = query.filter(Event.camera_id == camera_id)
            if start_date:
                query = query.filter(Event.timestamp >= start_date)
            if end_date:
                query = query.filter(Event.timestamp <= end_date)
            
            events = query.offset(offset).limit(limit).all()
            
            return [event.to_dict() for event in events]
            
        finally:
            session.close()
    
    def get_event_by_id(self, event_id: int) -> Optional[Dict]:
        """Get a single event by ID"""
        session = self.get_session()
        
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            return event.to_dict() if event else None
        finally:
            session.close()
    
    def delete_event(self, event_id: int) -> bool:
        """Delete an event by ID"""
        session = self.get_session()
        
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            if event:
                session.delete(event)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def mark_email_sent(self, event_id: int) -> bool:
        """Mark an event as having email sent"""
        session = self.get_session()
        
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            if event:
                event.email_sent = True
                event.email_sent_at = datetime.utcnow()
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def get_event_count(
        self,
        activity_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> int:
        """Get total count of events with optional filters"""
        session = self.get_session()
        
        try:
            query = session.query(func.count(Event.id))
            
            if activity_type:
                query = query.filter(Event.activity_type == activity_type)
            if start_date:
                query = query.filter(Event.timestamp >= start_date)
            if end_date:
                query = query.filter(Event.timestamp <= end_date)
            
            return query.scalar() or 0
            
        finally:
            session.close()
    
    # ==================== Statistics Operations ====================
    
    def get_statistics(self, days: int = 7) -> Dict:
        """
        Get detection statistics for the past N days
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary with statistics
        """
        session = self.get_session()
        start_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            # Total events
            total = session.query(func.count(Event.id)).filter(
                Event.timestamp >= start_date
            ).scalar() or 0
            
            # Events by type
            by_type = dict(
                session.query(Event.activity_type, func.count(Event.id))
                .filter(Event.timestamp >= start_date)
                .group_by(Event.activity_type)
                .all()
            )
            
            # Events by severity
            by_severity = dict(
                session.query(Event.severity, func.count(Event.id))
                .filter(Event.timestamp >= start_date)
                .group_by(Event.severity)
                .all()
            )
            
            # Events by day
            daily_counts = (
                session.query(
                    func.date(Event.timestamp).label('date'),
                    func.count(Event.id).label('count')
                )
                .filter(Event.timestamp >= start_date)
                .group_by(func.date(Event.timestamp))
                .all()
            )
            
            by_day = {str(row.date): row.count for row in daily_counts}
            
            # Recent events
            recent = session.query(Event).order_by(desc(Event.timestamp)).limit(10).all()
            
            return {
                'total_events': total,
                'by_type': by_type,
                'by_severity': by_severity,
                'by_day': by_day,
                'recent_events': [e.to_dict() for e in recent],
                'period_days': days
            }
            
        finally:
            session.close()
    
    # ==================== Camera Operations ====================
    
    def add_camera(
        self,
        camera_id: str,
        name: str,
        source: str,
        source_type: str = 'file',
        enabled: bool = True
    ) -> Camera:
        """Add a new camera configuration"""
        session = self.get_session()
        
        try:
            camera = Camera(
                id=camera_id,
                name=name,
                source=source,
                source_type=source_type,
                enabled=enabled
            )
            session.add(camera)
            session.commit()
            session.refresh(camera)
            return camera
        finally:
            session.close()
    
    def get_cameras(self, enabled_only: bool = False) -> List[Dict]:
        """Get all camera configurations"""
        session = self.get_session()
        
        try:
            query = session.query(Camera)
            if enabled_only:
                query = query.filter(Camera.enabled == True)
            
            cameras = query.all()
            return [cam.to_dict() for cam in cameras]
        finally:
            session.close()
    
    # ==================== Settings Operations ====================
    
    def set_setting(self, key: str, value: str):
        """Set a configuration value"""
        session = self.get_session()
        
        try:
            setting = session.query(AlertSettings).filter(AlertSettings.key == key).first()
            if setting:
                setting.value = value
            else:
                setting = AlertSettings(key=key, value=value)
                session.add(setting)
            session.commit()
        finally:
            session.close()
    
    def get_setting(self, key: str, default: str = None) -> Optional[str]:
        """Get a configuration value"""
        session = self.get_session()
        
        try:
            setting = session.query(AlertSettings).filter(AlertSettings.key == key).first()
            return setting.value if setting else default
        finally:
            session.close()
    
    def clear_old_events(self, days: int = 30) -> int:
        """
        Delete events older than N days
        
        Args:
            days: Delete events older than this many days
            
        Returns:
            Number of events deleted
        """
        session = self.get_session()
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        try:
            count = session.query(Event).filter(Event.timestamp < cutoff).delete()
            session.commit()
            return count
        finally:
            session.close()


# Test function
if __name__ == '__main__':
    # Initialize database
    db = Database('data/surveillance.db')
    
    # Log some test events
    event = db.log_event(
        activity_type='violence',
        severity='high',
        confidence=0.85,
        description='Test violence detection'
    )
    print(f"Created event: {event}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Statistics: {stats}")
    
    # Get events
    events = db.get_events(limit=10)
    print(f"Recent events: {events}")
