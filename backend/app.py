from flask import Flask, request, jsonify, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from authlib.integrations.flask_client import OAuth
from datetime import datetime
from dotenv import load_dotenv
import logging
import os
from bson import ObjectId

from services.auth_service import AuthService
from services.video_service import VideoService
from services.support_service import SupportService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

# App secret
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 500 * 1024 * 1024))

# MongoDB connection
try:
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client.snipx
    client.server_info()
    logger.info("✅ Connected to MongoDB")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {str(e)}")
    raise

# Initialize services
auth_service = AuthService(db)
video_service = VideoService(db)
support_service = SupportService(db)

# OAuth setup
oauth = OAuth(app)

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)



@app.route('/api/auth/google/login')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/api/auth/google/callback')
def google_callback():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token, nonce=None)

    user = db.users.find_one({'email': user_info['email']})
    if not user:
        user_id = str(db.users.insert_one({
            'email': user_info['email'],
            'first_name': user_info.get('given_name'),
            'last_name': user_info.get('family_name'),
            'oauth_id': user_info['sub'],
            'provider': 'google',
            'created_at': datetime.utcnow()
        }).inserted_id)
    else:
        user_id = str(user['_id'])

    jwt_token = auth_service.generate_token(user_id)
    return redirect(f"http://localhost:5173/auth/callback?token={jwt_token}")



def require_auth(f):
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401

        try:
            token = auth_header.split(' ')[1]
            user_id = auth_service.verify_token(token)
            return f(user_id, *args, **kwargs)
        except Exception as e:
            return jsonify({'error': str(e)}), 401

    decorated.__name__ = f.__name__
    return decorated

@app.route('/api/test-db', methods=['GET'])
def test_db():
    try:
        test_result = db.users.find_one()
        if test_result and '_id' in test_result:
            test_result['_id'] = str(test_result['_id'])

        return jsonify({
            "status": "success",
            "message": "MongoDB is connected",
            "sample_user": test_result
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "MongoDB connection failed",
            "error": str(e)
        }), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if not all(field in data for field in ['email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400

        user_id = auth_service.register_user(
            email=data['email'],
            password=data['password'],
            first_name=data.get('firstName'),
            last_name=data.get('lastName')
        )
        return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Signup error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or not all(field in data for field in ['email', 'password']):
            return jsonify({'message': 'Missing credentials'}), 400

        token, user = auth_service.login_user(data['email'], data['password'])
        return jsonify({'token': token, 'user': user}), 200
    except ValueError as e:
        return jsonify({'message': str(e)}), 401
    except Exception as e:
        logger.exception("Login error")
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/demo', methods=['POST'])
def demo_login():
    try:
        token, user = auth_service.create_demo_user()
        return jsonify({'token': token, 'user': user}), 200
    except Exception as e:
        logger.exception("Demo login error")
        return jsonify({'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        conversation_history = data.get('history', [])
        
        # Simple fallback response for now
        response = "I'm a basic chatbot. For detailed support, please use the support ticket system in the Help section."
        
        return jsonify({
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets', methods=['POST'])
@require_auth
def create_support_ticket(user_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['name', 'email', 'subject', 'description', 'priority', 'type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        ticket_id = support_service.create_ticket(user_id, data)
        
        return jsonify({
            'message': 'Support ticket created successfully',
            'ticket_id': ticket_id
        }), 201
        
    except Exception as e:
        logger.exception("Support ticket creation error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets', methods=['GET'])
@require_auth
def get_support_tickets(user_id):
    try:
        # Get tickets for the authenticated user
        tickets = support_service.get_user_tickets(user_id)
        return jsonify(tickets), 200
    except Exception as e:
        logger.exception("Get support tickets error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets/<ticket_id>', methods=['GET'])
@require_auth
def get_support_ticket(user_id, ticket_id):
    try:
        ticket = support_service.get_ticket(ticket_id)
        if not ticket:
            return jsonify({'error': 'Ticket not found'}), 404
        
        # Ensure user owns the ticket
        if str(ticket.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access to ticket'}), 403
            
        return jsonify(ticket.to_dict()), 200
    except Exception as e:
        logger.exception("Get support ticket error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_video(user_id):
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        video_id = video_service.save_video(file, user_id)

        return jsonify({'message': 'Video uploaded successfully', 'video_id': str(video_id)}), 200
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/process', methods=['POST'])
@require_auth
def process_video(user_id, video_id):
    try:
        options = request.json.get('options', {})
        video_service.process_video(video_id, options)
        return jsonify({'message': 'Processing completed successfully'}), 200
    except Exception as e:
        logger.error(f"Process error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>', methods=['GET'])
@require_auth
def get_video_status(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404

        # Convert custom Video object to dict
        if hasattr(video, 'to_dict'):
            video_dict = video.to_dict()
        elif hasattr(video, '__dict__'):
            video_dict = video.__dict__
        else:
            raise ValueError("Cannot serialize Video object")

        # Clean up any non-serializable fields (e.g., ObjectId)
        if '_id' in video_dict:
            video_dict['_id'] = str(video_dict['_id'])
        if 'user_id' in video_dict:
            video_dict['user_id'] = str(video_dict['user_id'])

        return jsonify(video_dict), 200

    except Exception as e:
        logger.error(f"Fetch video error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos', methods=['GET'])
@require_auth
def get_user_videos(user_id):
    try:
        videos = video_service.get_user_videos(user_id)
        return jsonify(videos), 200
    except Exception as e:
        logger.error(f"List videos error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>', methods=['DELETE'])
@require_auth
def delete_video(user_id, video_id):
    try:
        video_service.delete_video(video_id, user_id)
        return jsonify({'message': 'Video deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Delete video error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Add download endpoint for processed videos
@app.route('/api/videos/<video_id>/download', methods=['GET'])
@require_auth
def download_video(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get the processed video path
        processed_path = video.outputs.get('processed_video', video.filepath)
        
        if not os.path.exists(processed_path):
            return jsonify({'error': 'Processed video not found'}), 404
        
        return send_file(
            processed_path,
            as_attachment=True,
            download_name=f"enhanced_{video.filename}"
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/thumbnails/generate', methods=['POST'])
@require_auth
def generate_thumbnails(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        count = data.get('count', 5)
        style = data.get('style', 'auto')
        
        # Generate thumbnails
        video_service._generate_thumbnail(video)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Thumbnails generated successfully',
            'thumbnails': video.outputs.get('thumbnails', []),
            'count': len(video.outputs.get('thumbnails', []))
        }), 200
        
    except Exception as e:
        logger.error(f"Generate thumbnails error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/audio/enhance', methods=['POST'])
@require_auth
def enhance_audio_realtime(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        enhancement_type = data.get('type', 'full')
        noise_reduction = data.get('noiseReduction', True)
        volume_boost = data.get('volumeBoost', 20)
        
        # Enhanced audio processing options
        options = {
            'audio_enhancement_type': enhancement_type,
            'noise_reduction': noise_reduction,
            'volume_boost': volume_boost,
            'enhance_audio': True
        }
        
        video_service._enhance_audio(video, options)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Audio enhanced successfully',
            'enhancement_type': enhancement_type,
            'processed_audio': video.outputs.get('processed_video')
        }), 200
        
    except Exception as e:
        logger.error(f"Audio enhancement error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/status', methods=['GET'])
@require_auth
def get_processing_status(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Return detailed processing status
        status_info = {
            'status': video.status,
            'progress': {
                'upload': 100 if video.status != 'uploading' else 50,
                'processing': 100 if video.status == 'completed' else (75 if video.status == 'processing' else 0),
                'thumbnails': 100 if video.outputs.get('thumbnails') else 0,
                'audio_enhancement': 100 if video.outputs.get('processed_video') else 0,
                'subtitles': 100 if video.outputs.get('subtitles') else 0
            },
            'outputs': {
                'thumbnails_count': len(video.outputs.get('thumbnails', [])),
                'has_enhanced_audio': bool(video.outputs.get('processed_video')),
                'has_subtitles': bool(video.outputs.get('subtitles')),
                'has_summary': bool(video.outputs.get('summary'))
            },
            'metadata': video.metadata,
            'processing_time': (
                (video.process_end_time - video.process_start_time).total_seconds()
                if video.process_start_time and video.process_end_time
                else None
            )
        }
        
        return jsonify(status_info), 200
        
    except Exception as e:
        logger.error(f"Get processing status error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles', methods=['GET'])
@require_auth
def get_video_subtitles(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        subtitles_info = video.outputs.get('subtitles', {})
        if not subtitles_info:
            return jsonify([]), 200
        
        json_path = subtitles_info.get('json')
        if not json_path or not os.path.exists(json_path):
            return jsonify([]), 200
        
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)
        
        return jsonify(subtitle_data.get('segments', [])), 200
        
    except Exception as e:
        logger.error(f"Get subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles/<language>/download', methods=['GET'])
@require_auth
def download_subtitles(user_id, video_id, language):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        format_type = request.args.get('format', 'srt')
        subtitles_info = video.outputs.get('subtitles', {})
        
        if not subtitles_info:
            return jsonify({'error': 'No subtitles found'}), 404
        
        # If it's a string (old format), try to find the file
        if isinstance(subtitles_info, str):
            subtitle_path = subtitles_info
        else:
            subtitle_path = subtitles_info.get('srt' if format_type == 'srt' else 'json')
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            return jsonify({'error': 'Subtitle file not found'}), 404
        
        filename = f"{video.filename}_{language}.{format_type}"
        return send_file(
            subtitle_path,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Download subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles/generate', methods=['POST'])
@require_auth
def generate_subtitles(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        language = data.get('language', 'en')
        style = data.get('style', 'clean')
        
        # Generate subtitles
        options = {
            'subtitle_language': language,
            'subtitle_style': style,
            'generate_subtitles': True
        }
        
        video_service._generate_subtitles(video, options)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Subtitles generated successfully',
            'language': language,
            'style': style
        }), 200
        
    except Exception as e:
        logger.error(f"Generate subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

# Get current user info from JWT
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            user_id = data['user_id']
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 401
        return f(user_id, *args, **kwargs)
    return decorated

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user['_id'] = str(user['_id'])
    user.pop('password', None)
    return jsonify(user), 200

@app.route('/api/auth/delete-account', methods=['DELETE'])
@require_auth
def delete_account(user_id):
    try:
        # Delete user account and all associated data
        auth_service.delete_user(user_id)
        
        logger.info(f"User account deleted successfully: {user_id}")
        return jsonify({
            'message': 'Account deleted successfully',
            'deleted_user_id': user_id
        }), 200
        
    except ValueError as e:
        logger.error(f"Delete account error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Unexpected error deleting account for user {user_id}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)