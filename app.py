
"""
Main entry point for the AI Video-to-Text Converter application.

Authors: Bai Blyden, Darel Johnson
"""


import argparse
from modules.pipeline import VideoToTextPipeline
from modules.config import AppConfig



def main():
    parser = argparse.ArgumentParser(description="AI Video-to-Text Converter")
    parser.add_argument('--realtime', action='store_true', help='Enable real-time audio processing from microphone')
    args = parser.parse_args()

    config = AppConfig(realtime=args.realtime)
    pipeline = VideoToTextPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
