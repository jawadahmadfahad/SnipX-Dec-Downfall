from datetime import datetime
from bson.objectid import ObjectId

class SupportTicket:
    def __init__(self, user_id=None, name=None, email=None, subject=None, 
                 description=None, priority='medium', ticket_type='bug', 
                 status='open', created_at=None, updated_at=None, _id=None):
        self._id = ObjectId(_id) if _id else ObjectId()
        self.user_id = ObjectId(user_id) if user_id else None
        self.name = name
        self.email = email
        self.subject = subject
        self.description = description
        self.priority = priority  # low, medium, high, urgent
        self.type = ticket_type   # bug, feature, question, other
        self.status = status      # open, in_progress, resolved, closed
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.responses = []       # List of admin responses

    def to_dict(self):
        return {
            '_id': self._id,
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'subject': self.subject,
            'description': self.description,
            'priority': self.priority,
            'type': self.type,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'responses': self.responses
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            _id=data.get('_id'),
            user_id=data.get('user_id'),
            name=data.get('name'),
            email=data.get('email'),
            subject=data.get('subject'),
            description=data.get('description'),
            priority=data.get('priority', 'medium'),
            ticket_type=data.get('type', 'bug'),
            status=data.get('status', 'open'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
