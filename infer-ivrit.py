import faster_whisper
import requests
import tempfile
import os

# Load the faster-whisper model that supports Hebrew
model = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d4")

# URL of the audio file (replace this with the actual URL of your audio)
audio_url = "https://raw.githubusercontent.com/AshDavid12/runpod-serverless-forked/main/test_hebrew.wav"

# Download the audio file from the URL
response = requests.get(audio_url)
if response.status_code != 200:
    raise Exception("Failed to download audio file")

# Create a temporary file to store the audio
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
    tmp_audio_file.write(response.content)
    tmp_audio_file_path = tmp_audio_file.name

# Perform the transcription
segments, info = model.transcribe(tmp_audio_file_path, language="he")

# Print transcription results
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# Clean up the temporary file
os.remove(tmp_audio_file_path)


















