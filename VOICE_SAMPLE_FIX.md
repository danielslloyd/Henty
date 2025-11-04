# Voice Sample Format Fix - Testing Instructions

## The Problem We Fixed

When you tried to generate audio with a voice sample (Dan_Test1.wav), you got this error:
```
soundfile.LibsndfileError: Error opening 'voice_samples\Dan_Test1.wav': Format not recognised.
audioread.exceptions.NoBackendError
```

This happened because the recorded/uploaded voice sample wasn't in a format that Chatterbox TTS could read.

## The Solution

I've updated the server to **automatically convert all uploaded/recorded voice samples** to a proper WAV format that Chatterbox can read. The conversion happens in the upload endpoint using librosa and scipy.

## How to Test the Fix

### Step 1: Delete the Old Voice Sample

The existing `Dan_Test1.wav` file was saved in the old format. You need to delete it:

1. Navigate to: `c:\Users\danie\Desktop\Git\Henty\voice_samples\`
2. Delete the file `Dan_Test1.wav`

### Step 2: Restart the Server

1. Close the current server (Ctrl+C in the terminal)
2. Run `start_server.bat` again
3. Wait for the message: `Running on http://0.0.0.0:5000`

### Step 3: Re-upload or Re-record Your Voice Sample

Open `index.html` in your browser and either:

**Option A - Re-upload:**
1. Click the "Upload Voice Sample" button
2. Select your original voice recording file
3. Keep the name "Dan_Test1" (or choose a new name)
4. Click "Save"

**Option B - Re-record:**
1. Click "Record Voice Sample"
2. Allow microphone access
3. Click "Start Recording"
4. Speak for a few seconds
5. Click "Stop Recording"
6. Name it "Dan_Test1" (or choose a new name)
7. Click "Save"

### Step 4: Generate Audio with Voice Cloning

1. Browse to your folder with txt files
2. Expand one of the text file bubbles
3. Select your voice sample from the "Voice Sample" dropdown
4. Adjust the parameters if desired (Language, Exaggeration, CFG Weight)
5. Click "Generate New Audio"

### Step 5: Check the Results

**If it works:**
- You'll see "Audio generated successfully!" message
- A new audio file will appear in the collapsed section at the top of the bubble
- Playing it should sound like your voice

**If it still fails:**
- Check the server terminal for error messages
- Copy the entire error section and let me know
- Check the browser console (F12) for any errors

## What Changed in the Code

In `server.py`, the `/api/voice-samples/upload` endpoint now:

1. Saves the uploaded file temporarily
2. Uses `librosa.load()` to read the audio (supports many formats)
3. Converts the audio to int16 format
4. Saves it as a standard WAV file using `scipy.wavfile.write()`
5. Deletes the temporary file

This ensures that **regardless of what format you upload or record**, it gets converted to a format that Chatterbox TTS can read.

## Expected Behavior

- **Upload/Record:** Should show "Voice sample uploaded successfully"
- **Audio Generation:** Should complete without NoBackendError
- **Voice Cloning:** Generated audio should have characteristics of your voice sample

## Still Getting Errors?

If you still get the NoBackendError or any other error after following these steps, please provide:

1. The complete server terminal output (especially the error section)
2. The browser console output (F12 → Console tab)
3. Whether you re-uploaded or re-recorded the voice sample
4. The exact error message you see

I'll help debug further if needed!
