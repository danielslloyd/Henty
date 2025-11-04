# Quick Testing Checklist

## Voice Sample Fix - What You Need to Do

### ✅ Step-by-Step

1. **Delete old voice sample:**
   - Go to `voice_samples` folder
   - Delete `Dan_Test1.wav`

2. **Restart server:**
   - Close current server (Ctrl+C)
   - Run `start_server.bat`

3. **Re-upload voice sample:**
   - Open `index.html` in browser
   - Click "Upload Voice Sample" or "Record Voice Sample"
   - Save it (the server will automatically convert it to proper format)

4. **Test audio generation:**
   - Browse to txt files folder
   - Expand a text file bubble
   - Select your voice sample from dropdown
   - Click "Generate New Audio"

5. **Check results:**
   - Should see success message
   - Audio should appear in the bubble
   - Play it to verify voice cloning worked

## Expected Output

**Server terminal should show:**
```
Converting audio to proper WAV format...
Voice sample converted and saved: voice_samples\Dan_Test1.wav
```

**Browser should show:**
```
Voice sample uploaded successfully!
Audio generated successfully!
```

## If It Works

Great! Voice cloning is now working. You can:
- Upload multiple voice samples
- Generate audio with different voices
- Compare different parameter settings
- All voice samples will be automatically converted to the correct format

## If It Fails

Copy and send me:
1. Server terminal error output
2. Browser console errors (F12)
3. Whether you uploaded or recorded

I'll help debug!

---

For detailed explanation, see [VOICE_SAMPLE_FIX.md](VOICE_SAMPLE_FIX.md)
