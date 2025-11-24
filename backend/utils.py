import os
from werkzeug.utils import secure_filename
import magic

def get_mime_type(file_path):
    """Get MIME type of a file"""
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)

def is_valid_video(file_path):
    """Check if file is a valid video"""
    mime_type = get_mime_type(file_path)
    return mime_type.startswith('video/')

def create_unique_filename(filename):
    """Create unique filename to avoid conflicts"""
    name, ext = os.path.splitext(secure_filename(filename))
    counter = 1
    while os.path.exists(f"uploads/{name}_{counter}{ext}"):
        counter += 1
    return f"{name}_{counter}{ext}"