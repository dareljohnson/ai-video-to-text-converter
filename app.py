"""
Main entry point for the AI Video-to-Text Converter application.
"""

from modules.pipeline import VideoToTextPipeline
from modules.config import AppConfig


def main():
    config = AppConfig()
    pipeline = VideoToTextPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
