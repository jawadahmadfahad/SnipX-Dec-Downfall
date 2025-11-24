import { useState } from 'react';
import { Upload, CheckCircle, Loader } from 'lucide-react';

interface UploadSectionProps {
  setShowEditor: (show: boolean) => void;
}

const UploadSection = ({ setShowEditor }: UploadSectionProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    // Simulate file upload
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      simulateUpload();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      simulateUpload();
    }
  };

  const simulateUpload = () => {
    setIsUploading(true);
    setUploadProgress(0);
    
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setShowEditor(true);
          }, 500);
          return 100;
        }
        return newProgress;
      });
    }, 300);
  };

  return (
    <section id="try-it" className="py-16 px-4 bg-gradient-to-b from-white to-purple-50">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Try SnipX Now</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload your video to experience the power of automated video enhancement and summarization.
          </p>
        </div>
        
        <div 
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center
                    transition-colors duration-300 bg-white
                    ${isDragging ? 'border-purple-600 bg-purple-50' : 'border-gray-300'}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isUploading ? (
            <div className="w-full max-w-md">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Uploading video...</span>
                <span className="text-sm font-medium text-purple-700">{Math.round(uploadProgress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-purple-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              {uploadProgress === 100 && (
                <div className="flex items-center justify-center mt-4 text-green-600 gap-2">
                  <CheckCircle size={20} />
                  <span>Upload complete! Redirecting to editor...</span>
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="mb-4 p-4 rounded-full bg-purple-100">
                <Upload className="text-purple-600" size={32} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-900">Upload Your Video</h3>
              <p className="text-gray-600 mb-6 text-center max-w-md">
                Drag and drop your video file here, or click the button below to select from your device.
              </p>
              <label className="bg-purple-600 text-white px-6 py-3 rounded-md hover:bg-purple-700 
                              transition-colors cursor-pointer font-medium">
                Select Video
                <input 
                  type="file" 
                  className="hidden" 
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </label>
              <p className="mt-4 text-sm text-gray-500">
                Supported formats: MP4, MOV, AVI, MKV (Max 500MB)
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
};

export default UploadSection;