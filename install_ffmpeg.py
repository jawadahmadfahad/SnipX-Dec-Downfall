# FFmpeg Installation Guide for Windows

# For Windows, you can install ffmpeg using chocolatey or download directly

# Option 1: Using Chocolatey (if you have it installed)
# Open PowerShell as Administrator and run:
# choco install ffmpeg

# Option 2: Download and install manually
# 1. Go to https://ffmpeg.org/download.html
# 2. Download the Windows build
# 3. Extract to C:\ffmpeg
# 4. Add C:\ffmpeg\bin to your system PATH

# Option 3: Using winget (Windows Package Manager)
# Open PowerShell and run:
# winget install ffmpeg

# To verify installation, run:
# ffmpeg -version

print("FFmpeg installation instructions:")
print("1. Install using winget: winget install ffmpeg")
print("2. Or download from https://ffmpeg.org/download.html")
print("3. Add to system PATH")
print("4. Restart your terminal/IDE after installation")
