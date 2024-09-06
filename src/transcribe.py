import whisper
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

# Size	  Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
# tiny	  39 M	        tiny.en	            tiny	            ~1 GB	        ~32x
# base	  74 M	        base.en	            base	            ~1 GB	        ~16x
# small	  244 M	        small.en	        small	            ~2 GB	        ~6x
# medium  769 M	        medium.en	        medium	            ~5 GB	        ~2x
# large	  1550 M	    N/A	                large	            ~10 GB	         1x

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file into text using OpenAI's Whisper model.
    
    Supported Audio Formats:
    [MP3 (.mp3), WAV (.wav), FLAC (.flac), M4A (.m4a), OGG (.ogg), MP4 (.mp4), WEBM (.webm), MOV (.mov)]
    
    Parameters:
    audio_file_path (str): The path to the audio file.
    
    Returns:
    str: The transcribed text.
    """
    model = whisper.load_model("base")
    
    return model.transcribe(audio_file_path).get("text", "")

# if __name__ == "__main__":
#     audio_files = {
#         "M4A": r"Data\test.m4a",
#         "MP3": r"Data\test.mp3",
#         "MP4": r"Data\test.mp4",
#         "MPG": r"Data\test.mpg",
#         "WAV": r"Data\test.wav"
#     }
    
#     results = []
    
#     for format_name, file_path in audio_files.items():
#         try:
#             transcription = transcribe_audio(file_path)
#             results.append((format_name, transcription))
#         except Exception as e:
#             print(f"An error occurred while processing {format_name}: {e}")
    
#     for format_name, transcription in results:
#         print(f"{format_name}: {transcription}")