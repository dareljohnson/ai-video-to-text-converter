# AI Video-to-Text Converter

**Authors:** Bai Blyden, Darel Johnson

This project provides a robust, production-ready pipeline for converting audio or video files to text and subtitles using OpenAI Whisper, HuggingFace Transformers, and Python. It supports multiple languages, GPU acceleration, MFCC extraction, punctuation/capitalization correction, real-time transcription, and more.

![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)


**Latest Updates:**

- Added automatic directory creation and management
- Improved error handling for file operations
- Enhanced cross-platform path compatibility
- Added comprehensive test coverage (35 tests)
- Fixed repository size and management issues
- Improved model cache handling

---

## Features

### Input/Output Support

- Accepts audio/video files (MP4, AVI, MOV, MKV, WAV, MP3, FLAC, OGG)
- Batch processing of multiple files
- Real-time audio transcription from microphone
- Outputs plain text and SRT subtitles
- Organized output directory structure

### Transcription Capabilities

- Uses OpenAI Whisper Large V3 via HuggingFace
- Multi-language support with auto-detection
- Intelligent punctuation and capitalization correction
- Speaker identification and diarization
- Paragraph segmentation based on speech patterns

### Analysis & Features

- MFCC (Mel-frequency cepstral coefficients) extraction
- Speech pattern analysis (pauses, speaking rate)
- Word-level timeline tracking
- Transcription accuracy evaluation (WER)
- Detailed timing and progress tracking

### Technical Features

- GPU acceleration with CUDA support
- Cross-platform (Windows, Linux, macOS)
- Environment variable control for threading
- Comprehensive error handling
- Extensive unit test coverage with pytest
- Modular, extensible architecture
- CI/CD ready test suite

### Directory Structure

The program automatically manages the following directory structure:

```
ai-video-to-text-converter/
├── input/               # Place input audio/video files here
├── output/             # Generated transcripts and subtitles
│   ├── *.txt          # Text transcripts
│   └── *.srt          # Subtitle files
├── logs/              # Log files and processing records
├── model_cache/       # Downloaded models (gitignored)
└── tests/             # Test files and fixtures
```

Key features:
- Automatic directory creation
- Configurable output paths
- Cross-platform path handling
- Gitignored cache directories
- Organized output structure

---

## Directory Management

The program automatically handles all necessary directories:

1. **Standard Directories:**
   - `input/`: For input media files
   - `output/`: For transcripts and subtitles
   - `logs/`: For log files and reports
   - `model_cache/`: For downloaded models (auto-created as needed)

2. **Directory Creation:**
   - Creates missing directories automatically
   - Handles both file and directory paths
   - Creates parent directories as needed
   - Maintains proper permissions

3. **Path Handling:**
   - Cross-platform path compatibility
   - Relative path resolution
   - Absolute path conversion
   - Special character handling

---

## Contributing

We welcome contributions! Here's how you can help:

### Setting Up Development Environment

1. Fork and clone the repository
2. Create a virtual environment and install dependencies
3. Install development dependencies:

```sh
pip install -r requirements-dev.txt
```

### Running Tests

The project uses pytest for testing. Our test suite covers:

- Text Processing (WER, punctuation, formatting)
- Audio Processing (MFCC, speech patterns)
- System Features (logging, file operations)
- Memory Management
- Error Handling

To run tests:

```sh
# Run all tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_utils.py -v

# Run tests with coverage report
python -m pytest --cov=modules tests/
```

### Test Coverage

Current test coverage includes:

- Core Utilities (`test_utils.py`)
  - Text processing functions
  - Audio processing features
  - Configuration management
  - Error handling scenarios

- Pipeline Tests (`test_pipeline.py`)
  - End-to-end workflow testing
  - Batch processing validation
  - Memory management
  - Thread control
  - Device configuration

### Making Changes

1. Create a feature branch
2. Write tests for new features
3. Ensure all tests pass
4. Submit a pull request

---

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for audio/video processing)
- CUDA toolkit (optional, for GPU acceleration)

### Setup Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/dareljohnson/ai-video-to-text-converter.git
    cd ai-video-to-text-converter
    ```

2. **Create and activate a virtual environment:**

    ```sh
    # Create environment
    python -m venv ai-video-to-text_env

    # Activate on Windows PowerShell
    ./ai-video-to-text_env/Scripts/Activate.ps1
    
    # Activate on Windows CMD
    ai-video-to-text_env\Scripts\activate.bat
    
    # Activate on Linux/Mac
    source ai-video-to-text_env/bin/activate
    ```

3. **Install dependencies:**

All required packages are listed in `requirements.txt`. Install them with:

```sh
pip install -r requirements.txt
```

**IMPORTANT for GPU/CUDA users:**

The default `torch` in `requirements.txt` may be CPU-only. To use your NVIDIA GPU, uninstall torch and install the correct CUDA-enabled version for your CUDA toolkit:

For CUDA 12.1:
```sh
pip uninstall torch -y
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
For CUDA 11.8:
```sh
pip uninstall torch -y
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```
Check your CUDA version with `nvcc --version` or your NVIDIA control panel.

Verify GPU support:
```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA version:', torch.version.cuda); print('Torch version:', torch.__version__)"
```

**requirements.txt preview:**

```text
accelerate==1.10.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
async-timeout==5.0.1
attrs==25.3.0
audioread==3.0.1
certifi==2025.8.3
cffi==1.17.1
charset-normalizer==3.4.3
colorama==0.4.6
datasets==2.18.0
decorator==4.4.2
dill==0.3.8
filelock==3.18.0
frozenlist==1.7.0
fsspec==2024.2.0
git-filter-repo==2.47.0
huggingface-hub==0.34.4
idna==3.10
imageio==2.37.0
imageio-ffmpeg==0.6.0
Jinja2==3.1.6
joblib==1.5.1
lazy_loader==0.4
librosa==0.11.0
llvmlite==0.44.0
MarkupSafe==3.0.2
moviepy==1.0.3
mpmath==1.3.0
msgpack==1.1.1
multidict==6.6.4
multiprocess==0.70.16
networkx==3.4.2
numba==0.61.2
numpy==1.26.4
packaging==25.0
pandas==2.3.1
pillow==11.3.0
platformdirs==4.3.8
pooch==1.8.2
proglog==0.1.12
propcache==0.3.2
psutil==7.0.0
pyarrow==21.0.0
pyarrow-hotfix==0.7
pycparser==2.22
pydantic==2.11.7
pydantic_core==2.33.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
regex==2025.7.34
requests==2.32.4
safetensors==0.6.2
scikit-learn==1.7.1
scipy==1.15.3
six==1.17.0
sounddevice==0.5.2
soundfile==0.13.1
soxr==0.5.0.post1
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.19.1
torch==2.2.1
tqdm==4.67.1
transformers==4.40.0
typing-inspection==0.4.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
xxhash==3.5.0
yarl==1.20.1
```

4. **Install FFmpeg:**

   - Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system PATH.
   - Verify with:

   ```sh
   ffmpeg -version
   ```

---


## Usage

### Quick Start

1. **Place your media file in the `input` directory**
   - The directory will be created automatically if it doesn't exist
   - Supported formats: MP4, AVI, MOV, MKV, WAV, MP3, FLAC, OGG

2. **Activate the environment:**

    ```sh
    # Windows PowerShell
    ./ai-video-to-text_env/Scripts/Activate.ps1

    # Windows CMD
    ai-video-to-text_env\Scripts\activate.bat

    # Linux/Mac
    source ai-video-to-text_env/bin/activate
    ```

3. **Run the application:**

    ```sh
    python app.py
    ```

4. **Check the output directory**
   - Transcripts are saved in `output/output.txt`
   - Subtitles are saved in `output/output.srt`
   - Logs are saved in `logs/pipeline.log`

5. **Check outputs:**
   - Transcript: `output/output.txt`
   - Subtitles: `output/output.srt`
   - Timing details: `output/timings.txt`

### Batch Processing

Process all supported media files in the input directory:

```sh
python app.py --batch
```

Features:

- Processes all supported files in `input/` directory
- Generates unique output files for each input
- Saves timing information for each file
- Shows progress and status in console

### Real-Time Mode

Transcribe directly from your microphone:

```sh
python app.py --realtime
```

Features:

- Live transcription display in console
- Continuous saving to `output/realtime/realtime_transcript.txt`
- Optional MFCC extraction to `output/results/mfcc_chunk_*.npy`
- Speech pattern analysis to `output/results/patterns_chunk_*.json`
- Press Ctrl+C to stop recording

### Evaluating Accuracy

To calculate Word Error Rate (WER) between transcriptions:

```sh
python app.py --wer reference.txt hypothesis.txt
```

### Command Line Options

- `--realtime` : Enable microphone transcription
- `--batch` : Process all files in input directory
- `--cuda-device N` : Select GPU (default: 0)
- `--wer file1 file2` : Calculate Word Error Rate between files

```sh
python app.py --batch --no-pyannote
```

#### Example: Batch mode with pyannote diarization (default)

```sh
python app.py --batch
```

You can also control this in code or config (see below).

### Batch Mode

To transcribe all audio/video files in the `input` directory at once, use:

```sh
python app.py --batch
```

Each file will be transcribed and the outputs will be saved as `output/<filename>_output.txt` and `output/<filename>_output.srt`.

**Processing times for each file and the total batch are saved to `output/timings.txt`.**

---

## Advanced Features

### Transcription Accuracy (WER)

The Word Error Rate (WER) feature helps evaluate transcription accuracy:

```sh
# Generate ground truth transcript
echo "This is the correct transcript." > reference.txt

# Run transcription
python app.py
# This creates output/output.txt

# Compare with reference
python app.py --wer reference.txt output/output.txt
```

The WER score ranges from 0 (perfect) to 1 (completely different), helping you:

- Evaluate transcription quality
- Compare different model configurations
- Validate improvements in your pipeline

### Performance Analysis

Track detailed performance metrics:

1. **Per-file timing** in `output/timings.txt`:
   - Audio extraction time
   - ASR processing time
   - Feature extraction time
   - Total processing time

2. **Real-time analysis** during transcription:
   - Live word-level timestamps
   - Speech pattern detection
   - Speaker identification
   - MFCC visualization data

---

## Example

Suppose you have a file `meeting.mp4` in the project root:

```sh
python app.py
# Enter the RELATIVE path to an audio or video file in the project root: meeting.mp4
```

The transcribed text will be saved to `output/output.txt` and subtitles to `output/output.srt`.

---

## Integration Guide

Use the pipeline in your own Python projects:

### Basic Integration

```python
from modules.config import AppConfig
from modules.pipeline import VideoToTextPipeline

# Configure options
config = AppConfig(
    model="openai/whisper-large-v3",
    language="en",
    mfcc=True,
    punctuation_correction=True
)

# Initialize pipeline
pipeline = VideoToTextPipeline(config)

# Process a single file
pipeline.run()  # Will use config.input_path

# Or process multiple files
pipeline.run_batch()  # Will process all files in input directory
```

### Real-time Integration

```python
# Enable real-time processing
config = AppConfig(
    realtime=True,
    model="openai/whisper-large-v3",
    mfcc=True
)

pipeline = VideoToTextPipeline(config)
pipeline.run()  # Will start microphone capture
```

### Custom Processing

```python
# Customize output paths
config = AppConfig(
    input_path="my_videos/",
    output_text_path="results/transcript.txt",
    output_subs_path="results/subtitles.srt"
)

# Add timing callbacks
pipeline = VideoToTextPipeline(config)
pipeline.on_progress = lambda x: print(f"Progress: {x}%")
pipeline.run()

config = AppConfig(realtime=False)  # default is batch mode
pipeline = VideoToTextPipeline(config)
pipeline.run_batch()
```

### Real-Time Transcription Mode

For microphone input:

```python
from modules.config import AppConfig
from modules.pipeline import VideoToTextPipeline

config = AppConfig(realtime=True)
pipeline = VideoToTextPipeline(config)
pipeline.run()  # or pipeline.run_realtime()
```

You can also import and use utility functions from `modules.utils` for MFCC extraction, punctuation correction, etc.

---



## Customization

- Change model, language, HuggingFace token, thread limits, diarization method, or output paths by editing `modules/config.py` or passing parameters to `AppConfig`.
- Extend the pipeline or utility functions for advanced use cases.

---


## Configuration

### Model Settings

```python
config = AppConfig(
    # ASR Model
    model="openai/whisper-large-v3",
    language="en",  # or None for auto-detect
    
    # Processing Features
    mfcc=True,
    punctuation_correction=True,
    capitalization=True,
    track_timeline=True,
    
    # Speaker Recognition
    enable_speaker_recognition=True,
    min_speakers=1,
    max_speakers=10
)
```

### Hardware Optimization

```python
config = AppConfig(
    # GPU Settings
    device="cuda",  # or "cpu"
    cuda_device=0,  # GPU index
    
    # Thread Control
    omp_num_threads=6,
    mkl_num_threads=6,
    openblas_num_threads=6,
    numexpr_num_threads=6,
    torch_num_threads=6
)
```

### File Paths

```python
config = AppConfig(
    # Input/Output
    input_path="input/",  # Directory or file
    output_text_path="output/transcript.txt",
    output_subs_path="output/subtitles.srt",
    
    # Logging
    log_to_file=True,
    log_file="output/pipeline.log",
    show_progress=True
)
```

Thread limits help control CPU usage on shared or resource-limited systems. Adjust based on your hardware capabilities.



## Troubleshooting

### GPU Issues

If you have GPU-related problems:

1. **CUDA Setup**
   - Install compatible NVIDIA drivers
   - Install matching CUDA toolkit version
   - Verify with `nvidia-smi` command

2. **GPU Selection**
   - Use `--cuda-device N` to pick GPU
   - Check GPU index at startup message
   - Set `device="cpu"` to force CPU mode

3. **PyTorch Version**
   - Ensure PyTorch matches CUDA version
   - See installation section for details
   - Run provided verification script

### Common Issues

1. **No audio found**
   - Check input file exists and readable
   - Verify file format is supported
   - Check file permissions

2. **Slow processing**
   - Reduce thread limits in config
   - Use smaller model variant
   - Enable GPU acceleration

3. **Memory errors**
   - Process shorter audio segments
   - Reduce batch size
   - Free up system memory

---

## License

This project is licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the LICENSE file for the full license text.
