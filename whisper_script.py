import whisper
import sys
import json

def transcribe(audio_path):
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path, fp16=False, language="ja")
    return result

if __name__ == "__main__":
    audio_path = sys.argv[1]
    transcription = transcribe(audio_path)
    print(json.dumps(transcription))

