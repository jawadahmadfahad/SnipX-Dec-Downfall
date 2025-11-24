from datetime import datetime
from bson import ObjectId

class User:
    def __init__(self, email, password_hash, first_name=None, last_name=None):
        self.email = email
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.videos = []
        self.settings = {
            "default_language": "en",
            "auto_enhance_audio": False,
            "generate_thumbnails": True
        }

    def to_dict(self):
        return {
            "email": self.email,
            "password_hash": self.password_hash,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "videos": self.videos,
            "settings": self.settings
        }

    @staticmethod
    def from_dict(data):
        user = User(
            email=data["email"],
            password_hash=data.get("password_hash"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name")
        )
        user.created_at = data.get("created_at", datetime.utcnow())
        user.updated_at = data.get("updated_at", datetime.utcnow())
        user.videos = data.get("videos", [])
        user.settings = data.get("settings", {})
        return user
