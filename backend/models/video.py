from datetime import datetime
from bson import ObjectId

class Video:
    def __init__(self, user_id, filename, filepath, size):
        self.user_id = user_id
        self.filename = filename
        self.filepath = filepath
        self.size = size
        self.status = "uploaded"
        self.processing_options = {}
        self.upload_date = datetime.utcnow()
        self.process_start_time = None
        self.process_end_time = None
        self.error = None
        self.metadata = {
            "duration": None,
            "format": None,
            "resolution": None,
            "fps": None
        }
        self.outputs = {
            "processed_video": None,
            "thumbnail": None,
            "subtitles": None,
            "summary": None
        }

    def to_dict(self):
        return {
            "user_id": str(self.user_id),
            "filename": self.filename,
            "filepath": self.filepath,
            "size": self.size,
            "status": self.status,
            "processing_options": self.processing_options,
            "upload_date": self.upload_date,
            "process_start_time": self.process_start_time,
            "process_end_time": self.process_end_time,
            "error": self.error,
            "metadata": self.metadata,
            "outputs": self.outputs
        }

    @staticmethod
    def from_dict(data):
        video = Video(
            user_id=ObjectId(data["user_id"]),
            filename=data["filename"],
            filepath=data["filepath"],
            size=data["size"]
        )
        video.status = data.get("status", "uploaded")
        video.processing_options = data.get("processing_options", {})
        video.upload_date = data.get("upload_date", datetime.utcnow())
        video.process_start_time = data.get("process_start_time")
        video.process_end_time = data.get("process_end_time")
        video.error = data.get("error")
        video.metadata = data.get("metadata", {})
        video.outputs = data.get("outputs", {})
        return video