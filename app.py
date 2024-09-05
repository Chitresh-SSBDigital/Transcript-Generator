from flask import Flask, request, jsonify
from pathlib import Path
import logging
from uuid import uuid4
from src.transcribe import transcribe_audio

# Initialize Flask app
app = Flask(__name__)

# Define the log file name
log_file = 'log/app.log'

# Set up logging to the file, with no extra information in the log format
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def process_audio(file: Path) -> str:
    """
    Processes and transcribes an audio file.
    """
    if file.suffix not in "['.MP3', '.MP4', '.MPGA', '.M4A', '.WAV']":
        raise ValueError("Invalid file format. Supported formats are MP3, MP4, MPGA, M4A, and WAV.")
    
    try:
        transcription = transcribe_audio(str(file))
        
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        raise RuntimeError(f"An error occurred during transcription: {e}")
    
    # Optionally, remove the file after processing
    if file.exists():
        file.unlink()
    
    return transcription



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.mp3', '.mp4', '.wav', '.m4a', '.mpga')):
        try:
            # Save the file to a temporary location
            file_path = Path(f"temp_audio_file_{uuid4().hex}")
            file.save(file_path)
            
            # Process and transcribe the audio file
            transcription = process_audio(file_path)
            
            return jsonify({"transcription": transcription}), 200
        
        except ValueError as e:
            logger.error(f"ValueError: {e}")
            return jsonify({"error": str(e)}), 400
        
        except RuntimeError as e:
            logger.error(f"RuntimeError: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)