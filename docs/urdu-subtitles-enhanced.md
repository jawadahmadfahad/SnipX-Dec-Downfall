# Urdu Subtitle Generation - Enhanced with Whisper Large-v3

## 🎯 Overview

This system now uses **Whisper Large-v3**, the most accurate speech recognition model, specifically optimized for Urdu language transcription.

## 🚀 Key Improvements

### 1. **Best Whisper Model**
- **Urdu/Roman Urdu**: Uses `large-v3` model (highest accuracy)
- **Arabic/Hindi**: Uses `large` model
- **Other languages**: Optimized model selection

### 2. **Enhanced Audio Preprocessing**
For Urdu audio, the system now applies:
- Advanced noise reduction
- Dynamic range compression
- Frequency filtering (80Hz - 8000Hz)
- Volume normalization and boosting
- Optimal sampling rate (16kHz)

### 3. **Optimized Transcription Settings**
```python
{
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Multiple attempts
    "beam_size": 5,                             # Better accuracy
    "best_of": 5,                               # Sample multiple times
    "initial_prompt": "یہ ایک اردو زبان کی ویڈیو ہے۔ صاف اور درست الفاظ استعمال کریں۔"
}
```

### 4. **Post-Processing**
- Automatic Urdu punctuation fixing
- Removal of transcription artifacts
- Proper spacing and formatting

## 📦 Installation

### Step 1: Upgrade Whisper
```powershell
cd backend
python upgrade_whisper.py
```

This will:
- Remove old Whisper packages
- Install latest OpenAI Whisper
- Install required dependencies (librosa, scipy, numba)
- Verify installation

### Step 2: Install Dependencies (if needed)
```powershell
pip install -r requirements.txt
```

### Step 3: Download Whisper Models
The models will auto-download on first use, but you can pre-download:
```powershell
python -c "import whisper; whisper.load_model('large-v3')"
```

**Note**: The `large-v3` model is ~3GB and will take time to download on first use.

## 🎬 Usage

### Via API

```python
# Generate Urdu subtitles
POST /api/videos/{video_id}/subtitles/generate
{
    "language": "ur",        # or "ru-ur" for Roman Urdu
    "style": "clean"
}
```

### Language Codes
- `ur` - اردو (Urdu script)
- `ru-ur` - Roman Urdu (English script)

## ⚡ Performance

### Model Comparison for Urdu

| Model | Size | Speed | Accuracy | Recommended |
|-------|------|-------|----------|-------------|
| tiny | 39MB | Fast | Low | ❌ No |
| base | 74MB | Fast | Medium | ❌ No |
| small | 244MB | Medium | Good | ⚠️ Maybe |
| medium | 769MB | Slow | Better | ⚠️ Previous |
| large | 1.5GB | Slower | Excellent | ✅ Good |
| **large-v3** | **3GB** | **Slowest** | **Best** | ✅ **YES** |

### Processing Time (Approximate)
- 1-minute video: ~2-3 minutes
- 5-minute video: ~8-12 minutes
- 10-minute video: ~15-20 minutes

**Note**: Times vary based on:
- CPU/GPU availability
- Audio quality
- Speech clarity
- Background noise levels

## 🔧 Technical Details

### Audio Preprocessing Pipeline
1. **Normalize** - Ensure consistent volume levels
2. **Compress** - Apply dynamic range compression
3. **High-pass filter** - Remove low-frequency noise (< 80Hz)
4. **Low-pass filter** - Remove high-frequency noise (> 8kHz)
5. **Boost** - Add +3dB for better detection
6. **Mono conversion** - Ensure single channel
7. **Resample** - Convert to 16kHz (Whisper optimal)
8. **Final normalize** - Normalize after all processing

### Transcription Options
- **Multiple temperatures**: Try 5 different temperature settings
- **Beam search**: Use beam size of 5 for better accuracy
- **Best of 5**: Generate 5 samples and pick the best
- **Context conditioning**: Use previous text for context
- **Initial prompt**: Urdu language hint for better recognition
- **FP16**: Use half-precision for GPU acceleration

### Post-Processing
- Fix common Urdu punctuation issues
- Remove transcription artifacts
- Normalize whitespace
- Add proper sentence endings

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Download the model manually:
```powershell
python -c "import whisper; whisper.load_model('large-v3')"
```

### Issue: "Out of memory"
**Solution**: Use a smaller model:
```python
# Edit video_service.py, line 740
return "large"  # Instead of "large-v3"
```

### Issue: "Slow processing"
**Solutions**:
1. Use GPU if available (auto-detected)
2. Use smaller model for faster processing
3. Preprocess audio to remove silence first

### Issue: "Poor accuracy for specific words"
**Solution**: Use initial prompt with context:
```python
initial_prompt = "اس ویڈیو میں [specific topic] کے بارے میں بات ہو رہی ہے۔"
```

## 📊 Quality Metrics

The system now tracks:
- **Confidence scores** for each segment
- **Language detection** accuracy
- **Processing time** statistics
- **Segment count** and duration

## 🔮 Future Improvements

1. **Custom Urdu model**: Train domain-specific model
2. **Real-time transcription**: Streaming API support
3. **Speaker diarization**: Identify multiple speakers
4. **Word-level timestamps**: More precise timing
5. **Translation**: Auto-translate to/from Urdu

## 📚 References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md)
- [Whisper Performance](https://github.com/openai/whisper/discussions)

## 🤝 Support

For issues or questions:
1. Check the console logs for detailed debugging info
2. Verify audio quality (clear speech, minimal background noise)
3. Try preprocessing audio separately before transcription
4. Contact support with video details and error logs

---

**Last Updated**: November 2025  
**Whisper Version**: Large-v3 (Latest)  
**Optimized For**: Urdu, Roman Urdu, Arabic, Hindi
