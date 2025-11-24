import { useState, useRef, useEffect } from 'react';
import { Upload, Video, X, UploadCloud as CloudUpload, Scissors, Trash2, Undo2, Music, Type, Save, Magnet as Magic, Brain, Volume2 } from 'lucide-react';
import SubtitleEditor from './SubtitleEditor';
import VideoPlayer from './VideoPlayer';
import { ApiService } from '../services/api';
import toast from 'react-hot-toast';

interface VideoEditorProps {
  videoUrl?: string;
}

const VideoEditor = ({ videoUrl }: VideoEditorProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(100);
  const [activeTab, setActiveTab] = useState('cutting');
  const [showAudioSettings, setShowAudioSettings] = useState(false);
  const [showSubtitleEditor, setShowSubtitleEditor] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [processedVideoData, setProcessedVideoData] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(100);
  const [isTrimMode, setIsTrimMode] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      
      // Create object URL for preview
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      
      // Upload the file to backend
      await uploadFile(file);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      
      // Create object URL for preview
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      
      // Upload the file to backend
      await uploadFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const uploadFile = async (file: File) => {
    try {
      setUploadProgress(0);
      console.log('Uploading file:', file.name);
      
      const response = await ApiService.uploadVideo(file, (progress) => {
        setUploadProgress(progress);
      });
      
      console.log('Upload response:', response);
      setVideoId(response.video_id);
      toast.success('Video uploaded successfully!');
      
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('Upload failed. Please try again.');
    }
  };

  const handleProcessVideo = async () => {
    if (!videoId) {
      toast.error('Please upload a video first');
      return;
    }

    setIsProcessing(true);
    try {
      console.log('Processing video:', videoId);
      
      const options = {
        generate_subtitles: true,
        subtitle_language: 'en',
        subtitle_style: 'clean',
        cut_silence: true,
        enhance_audio: true
      };

      await ApiService.processVideo(videoId, options);
      
      // Get the processed video data
      const videoData = await ApiService.getVideo(videoId);
      console.log('Processed video data:', videoData);
      
      setProcessedVideoData(videoData);
      toast.success('Video processed successfully!');
      
    } catch (error) {
      console.error('Processing failed:', error);
      toast.error('Processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearVideo = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setVideoId(null);
    setProcessedVideoData(null);
    setUploadProgress(0);
    setIsProcessing(false);
    setTrimStart(0);
    setTrimEnd(100);
    setIsTrimMode(false);
  };

  const handleTrim = () => {
    if (!videoId) {
      toast.error('Please upload a video first');
      return;
    }
    setIsTrimMode(!isTrimMode);
    if (!isTrimMode) {
      toast.success('Trim mode activated. Adjust the timeline sliders.');
    }
  };

  const handleSave = async () => {
    if (!videoId) {
      toast.error('Please upload a video first');
      return;
    }

    setIsSaving(true);
    try {
      toast.success('Saving video with trim settings...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success('Video saved successfully!');
    } catch (error) {
      toast.error('Failed to save video');
      console.error('Save error:', error);
    } finally {
      setIsSaving(false);
    }
  };
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Cleanup object URL when component unmounts or when previewUrl changes
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-2xl overflow-hidden border border-white/20 animate-fade-in">
      <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-pink-50">
        <h2 className="text-2xl font-bold text-gray-800 animate-slide-in-left">Video Editor</h2>
        <p className="text-gray-600 mt-1 animate-slide-in-left" style={{ animationDelay: '0.1s' }}>Upload your video and let AI do the magic</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
        {/* Left Panel - Upload & Settings */}
        <div className="lg:col-span-1 space-y-6">
          {/* Upload Area */}
          <div
            className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center transition-all duration-300 hover:border-purple-400 hover:bg-purple-50/50 hover:shadow-lg transform hover:scale-105"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            style={{
              animation: 'fade-in-up 0.6s ease-out'
            }}
          >
            <div className="flex justify-center">
              <CloudUpload className="text-purple-600" size={40} />
            </div>
            <h3 className="mt-4 text-lg font-medium text-gray-900">Drag & Drop Video</h3>
            <p className="mt-1 text-sm text-gray-600">or click to browse files</p>
            <input 
              type="file" 
              className="hidden" 
              id="video-upload" 
              accept="video/*"
              onChange={handleFileChange}
            />
            <button
              onClick={() => document.getElementById('video-upload')?.click()}
              className="mt-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-6 py-3 rounded-xl text-sm font-medium transition-all duration-300 transform hover:scale-105 hover:shadow-xl"
              style={{
                boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)'
              }}
            >
              Select Video
            </button>
          </div>

          {/* Selected File Info */}
          {selectedFile && (
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center">
                <Video className="text-purple-600 mr-3" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
                <button className="text-gray-400 hover:text-gray-500">
                  <X size={20} />
                </button>
              </div>
              <div className="mt-3 bg-gray-200 rounded-full h-1.5">
                <div 
                  className="bg-purple-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="mt-1 text-xs text-gray-500 text-right">{uploadProgress}% uploaded</p>
            </div>
          )}

          {/* AI Tools Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">AI Editing Tools</h3>
            
            {/* Debug Test Button */}
            {videoId && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <button
                  onClick={async () => {
                    console.log('🧪 Testing subtitle API for videoId:', videoId);
                    try {
                      const subs = await ApiService.getVideoSubtitles(videoId);
                      console.log('🧪 Test result:', subs);
                      toast.success(`Found ${subs?.length || 0} subtitle segments`);
                    } catch (error) {
                      console.error('🧪 Test failed:', error);
                      toast.error('Subtitle API test failed: ' + error);
                    }
                  }}
                  className="w-full bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md text-sm transition-colors"
                >
                  🧪 Test Subtitle API (Debug)
                </button>
              </div>
            )}

            {/* Video Summarization */}
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Brain className="text-purple-600 mr-3" size={20} />
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Video Summarization</h4>
                    <p className="text-xs text-gray-500">Extract key moments automatically</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                </label>
              </div>
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Summary Length</span>
                  <span className="text-xs font-medium text-purple-600">30% of original</span>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="50" 
                  defaultValue="30"
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            {/* Auto Subtitles */}
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Type className="text-purple-600 mr-3" size={20} />
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Auto Subtitles</h4>
                    <p className="text-xs text-gray-500">Generate captions in multiple languages</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                </label>
              </div>
              <div className="mt-3 space-y-2">
                <label className="block text-xs text-gray-600">Language</label>
                <select className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 text-sm">
                  <option>English</option>
                  <option>Urdu (اردو)</option>
                  <option>Roman Urdu</option>
                  <option>Arabic (العربية)</option>
                  <option>Hindi (हिंदी)</option>
                  <option>Spanish</option>
                  <option>French</option>
                  <option>German</option>
                  <option>Chinese (中文)</option>
                  <option>Japanese</option>
                  <option>Korean (한국어)</option>
                  <option>Portuguese</option>
                  <option>Russian (Русский)</option>
                  <option>Italian</option>
                  <option>Turkish</option>
                  <option>Dutch</option>
                </select>
                <label className="block text-xs text-gray-600 mt-2">Style</label>
                <select className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 text-sm">
                  <option value="clean">Clean</option>
                  <option value="casual">Casual</option>
                  <option value="formal">Formal</option>
                  <option value="creative">Creative</option>
                </select>
              </div>
            </div>

            {/* Audio Enhancement */}
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Volume2 className="text-purple-600 mr-3" size={20} />
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Audio Enhancement</h4>
                    <p className="text-xs text-gray-500">Remove noise and balance levels</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    className="sr-only peer" 
                    onChange={(e) => setShowAudioSettings(e.target.checked)}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                </label>
              </div>
              {showAudioSettings && (
                <div className="mt-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">Noise Reduction</span>
                    <span className="text-xs font-medium text-purple-600">Medium</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="2" 
                    defaultValue="1"
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-gray-600">Volume Boost</span>
                    <span className="text-xs font-medium text-purple-600">20%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    defaultValue="20"
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Process Button */}
          <button
            onClick={handleProcessVideo}
            disabled={!videoId || isProcessing}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-400 text-white py-4 px-6 rounded-xl font-semibold flex items-center justify-center transition-all duration-300 transform hover:scale-105 hover:shadow-2xl disabled:hover:scale-100 disabled:cursor-not-allowed"
            style={{
              boxShadow: !isProcessing && videoId ? '0 8px 25px rgba(139, 92, 246, 0.5)' : 'none',
              animation: videoId && !isProcessing ? 'pulse-glow 2s ease-in-out infinite' : 'none'
            }}
          >
            <Magic className="mr-2" size={20} />
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              'Process Video'
            )}
          </button>
        </div>

        {/* Center Panel - Video Preview */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900 aspect-video rounded-lg overflow-hidden flex items-center justify-center relative">
            {!previewUrl ? (
              <div className="text-center text-gray-400">
                <Video className="mx-auto mb-3" size={48} />
                <p>Video preview will appear here</p>
              </div>
            ) : (
              <VideoPlayer
                videoUrl={previewUrl}
                videoId={videoId || undefined}
                subtitles={processedVideoData?.subtitles}
                onTimeUpdate={(time) => setCurrentTime(time)}
              />
            )}
          </div>

          {/* Timeline Editor */}
          <div className="mt-6 bg-gray-50 rounded-lg p-4 border-2 border-transparent transition-all duration-300" style={{
            borderColor: isTrimMode ? '#8b5cf6' : 'transparent',
            boxShadow: isTrimMode ? '0 0 20px rgba(139, 92, 246, 0.3)' : 'none'
          }}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-900 flex items-center gap-2">
                Timeline Editor
                {isTrimMode && (
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full animate-pulse">
                    Trim Mode Active
                  </span>
                )}
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={handleTrim}
                  className={`p-1 transition-all duration-300 transform hover:scale-110 ${
                    isTrimMode ? 'text-purple-600 bg-purple-100 rounded-md' : 'text-gray-500 hover:text-purple-600'
                  }`}
                  title="Trim Video"
                >
                  <Scissors size={20} />
                </button>
                <button
                  onClick={clearVideo}
                  className="p-1 text-gray-500 hover:text-red-600 transition-all duration-300 transform hover:scale-110"
                  title="Clear Video"
                >
                  <Trash2 size={20} />
                </button>
                <button className="p-1 text-gray-500 hover:text-purple-600 transition-all duration-300 transform hover:scale-110" title="Undo">
                  <Undo2 size={20} />
                </button>
              </div>
            </div>

            {isTrimMode && (
              <div className="mb-4 space-y-3 animate-slide-down">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-700">Trim Start: <span className="font-medium text-purple-600">{trimStart}%</span></span>
                  <span className="text-gray-700">Trim End: <span className="font-medium text-purple-600">{trimEnd}%</span></span>
                </div>
                <div className="space-y-2">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={trimStart}
                    onChange={(e) => setTrimStart(Math.min(Number(e.target.value), trimEnd - 1))}
                    className="w-full h-2 bg-gradient-to-r from-purple-200 to-purple-400 rounded-lg appearance-none cursor-pointer"
                  />
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={trimEnd}
                    onChange={(e) => setTrimEnd(Math.max(Number(e.target.value), trimStart + 1))}
                    className="w-full h-2 bg-gradient-to-r from-pink-200 to-pink-400 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            )}

            <div className="bg-white border border-gray-200 rounded-md p-2 h-20 overflow-x-auto relative">
              {isTrimMode && (
                <div
                  className="absolute top-0 bottom-0 bg-purple-600/20 border-x-2 border-purple-600 z-10 pointer-events-none transition-all duration-300"
                  style={{
                    left: `${trimStart}%`,
                    right: `${100 - trimEnd}%`
                  }}
                />
              )}
              <div className="flex items-center h-full min-w-max relative">
                {[24, 16, 32, 20, 28].map((width, index) => (
                  <div
                    key={index}
                    className="h-12 rounded-sm mx-1 border transition-all duration-300"
                    style={{
                      width: `${width * 4}px`,
                      backgroundColor: isTrimMode ? '#e9d5ff' : '#f3e8ff',
                      borderColor: isTrimMode ? '#a855f7' : '#d8b4fe',
                      transform: isTrimMode ? 'scaleY(1.05)' : 'scaleY(1)'
                    }}
                  />
                ))}
              </div>
            </div>

            <div className="mt-3 flex justify-between items-center">
              <div className="flex space-x-2">
                <button className="px-3 py-2 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 rounded-lg text-sm flex items-center transition-all duration-300 transform hover:scale-105 hover:shadow-md">
                  <Music className="mr-1" size={16} />
                  Add Music
                </button>
                <button className="px-3 py-2 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 rounded-lg text-sm flex items-center transition-all duration-300 transform hover:scale-105 hover:shadow-md">
                  <Type className="mr-1" size={16} />
                  Add Text
                </button>
                <button
                  onClick={() => setShowSubtitleEditor(true)}
                  disabled={!videoId}
                  className="px-3 py-2 bg-gradient-to-r from-purple-100 to-purple-200 hover:from-purple-200 hover:to-purple-300 text-purple-700 rounded-lg text-sm flex items-center transition-all duration-300 transform hover:scale-105 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  <Type className="mr-1" size={16} />
                  Edit Subtitles
                </button>
              </div>
              <button
                onClick={handleSave}
                disabled={!videoId || isSaving}
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg text-sm font-medium flex items-center transition-all duration-300 transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                style={{
                  boxShadow: !isSaving && videoId ? '0 4px 15px rgba(139, 92, 246, 0.4)' : 'none'
                }}
              >
                {isSaving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="mr-2" size={16} />
                    Save Project
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Custom Animations */}
      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slide-in-left {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes slide-down {
          from {
            opacity: 0;
            max-height: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            max-height: 200px;
            transform: translateY(0);
          }
        }

        @keyframes pulse-glow {
          0%, 100% {
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
          }
          50% {
            box-shadow: 0 8px 35px rgba(236, 72, 153, 0.6);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }

        .animate-slide-in-left {
          animation: slide-in-left 0.6s ease-out;
        }

        .animate-slide-down {
          animation: slide-down 0.4s ease-out;
        }
      `}</style>

      {/* Subtitle Editor Modal */}
      {showSubtitleEditor && videoId && (
        <SubtitleEditor
          videoId={videoId}
          onClose={() => setShowSubtitleEditor(false)}
          onSubtitlesUpdate={(subtitles) => {
            console.log('Subtitles updated:', subtitles);
          }}
        />
      )}
    </div>
  );
};

export default VideoEditor;