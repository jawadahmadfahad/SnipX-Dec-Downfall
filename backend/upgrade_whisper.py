"""
Script to upgrade Whisper to the latest version for best Urdu transcription accuracy.
This will install the latest OpenAI Whisper with large-v3 model support.
"""

import subprocess
import sys

def upgrade_whisper():
    """Upgrade Whisper to the latest version"""
    print("=" * 60)
    print("🔄 Upgrading Whisper to Latest Version for Best Urdu Support")
    print("=" * 60)
    
    # Uninstall old versions
    print("\n📦 Removing old Whisper packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "whisper"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "openai-whisper"], check=False)
    except Exception as e:
        print(f"⚠️  Warning during uninstall: {e}")
    
    # Install latest version
    print("\n📦 Installing latest OpenAI Whisper...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "openai-whisper"
        ], check=True)
        print("✅ OpenAI Whisper installed successfully!")
    except Exception as e:
        print(f"❌ Error installing Whisper: {e}")
        return False
    
    # Install/upgrade required dependencies
    print("\n📦 Installing/upgrading dependencies...")
    dependencies = [
        "librosa",
        "scipy",
        "numba",
        "ffmpeg-python"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", dep
            ], check=True)
            print(f"✅ {dep} installed/upgraded successfully!")
        except Exception as e:
            print(f"⚠️  Warning with {dep}: {e}")
    
    # Verify installation
    print("\n🔍 Verifying Whisper installation...")
    try:
        import whisper
        print(f"✅ Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")
        
        # List available models
        print("\n📋 Available Whisper models:")
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for model in models:
            print(f"   - {model}")
        
        print("\n" + "=" * 60)
        print("✅ Upgrade Complete!")
        print("=" * 60)
        print("\n📝 Next Steps:")
        print("   1. Restart your Flask application")
        print("   2. Upload a video with Urdu audio")
        print("   3. Select 'Urdu' or 'Roman Urdu' for subtitles")
        print("   4. The system will now use 'large-v3' model for best accuracy!")
        print("\n💡 Tips:")
        print("   - Large-v3 model will download on first use (~3GB)")
        print("   - Processing will be slower but much more accurate")
        print("   - Best for Urdu, Arabic, Hindi, and other complex languages")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Error verifying Whisper: {e}")
        return False

if __name__ == "__main__":
    success = upgrade_whisper()
    sys.exit(0 if success else 1)
