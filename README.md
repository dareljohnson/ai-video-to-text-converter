
# AI Video-to-Text Converter

This project provides a robust, production-ready pipeline for converting audio or video files to text and subtitles using OpenAI Whisper, HuggingFace Transformers, and Python. It supports multiple languages, GPU acceleration, MFCC extraction, punctuation/capitalization correction, and more.

---

## Features

- Accepts audio or video as input (MP4, AVI, MOV, MKV, WAV, etc.)
- Uses GPU if available (Torch, CUDA)
- Uses `openai/whisper-small` via HuggingFace ASR pipeline
- Multi-language support (auto-detect or specify language)
- Feature extraction (MFCCs)
- Punctuation and capitalization correction
- Timeline tracking and subtitle (SRT) generation
- Speech pattern analysis (pauses, speaking rate)
- Outputs plain text and subtitles to the `output/` directory
- Modular, extensible, and easy to integrate into other Python projects

---

## Installation

1. **Clone the repository:**
	```sh
	git clone https://github.com/dareljohnson/ai-video-to-text-converter.git
	cd ai-video-to-text-converter
	```

2. **Create and activate a Python 3.10 virtual environment:**
	```sh
	python -m venv ai-video-to-text_env
	# On Windows PowerShell:
	.\ai-video-to-text_env\Scripts\Activate.ps1
	# On Windows CMD:
	ai-video-to-text_env\Scripts\activate.bat
	```

3. **Install dependencies:**
	```sh
	pip install torch==2.2.1 transformers==4.40.0 datasets==2.18.0 moviepy==1.0.3 accelerate==0.30.1 librosa soundfile pydantic
	```

4. **Install FFmpeg:**
	- Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system PATH.
	- Verify with:
	  ```sh
	  ffmpeg -version
	  ```

---

## Usage

1. **Place your audio or video file in the project root directory.**
2. **Run the application:**
	```sh
	./ai-video-to-text_env/Scripts/Activate.ps1  # PowerShell
	python app.py
	```
3. **When prompted, enter the relative path to your file (e.g., `myfile.mp4`).**
4. **Output:**
	- Transcription: `output/output.txt`
	- Subtitles: `output/output.srt`

---

## Example

Suppose you have a file `meeting.mp4` in the project root:

```sh
python app.py
# Enter the RELATIVE path to an audio or video file in the project root: meeting.mp4
```

The transcribed text will be saved to `output/output.txt` and subtitles to `output/output.srt`.

---

## Integration Example

You can use the pipeline in your own Python projects:

```python
from modules.config import AppConfig
from modules.pipeline import VideoToTextPipeline

config = AppConfig(input_path="myfile.mp4")
pipeline = VideoToTextPipeline(config)
pipeline.run()
```

You can also import and use utility functions from `modules.utils` for MFCC extraction, punctuation correction, etc.

---

## Customization

- Change model, language, or output paths by editing `modules/config.py` or passing parameters to `AppConfig`.
- Extend the pipeline or utility functions for advanced use cases.

---

## Troubleshooting

- **FFmpeg not found:** Ensure FFmpeg is installed and in your PATH.
- **CUDA not used:** Make sure you have a compatible GPU and CUDA drivers.
- **FileNotFoundError:** Ensure your input file is in the project root and you provide the correct relative path.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
