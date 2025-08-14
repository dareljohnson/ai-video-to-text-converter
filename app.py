
"""
Main entry point for the AI Video-to-Text Converter application.

Authors: Bai Blyden, Darel Johnson
"""



import argparse
from modules.pipeline import VideoToTextPipeline
from modules.config import AppConfig
from modules.utils import compute_wer




def main():
    parser = argparse.ArgumentParser(description="AI Video-to-Text Converter")
    parser.add_argument('--realtime', action='store_true', help='Enable real-time audio processing from microphone')
    parser.add_argument('--wer', nargs=2, metavar=('REFERENCE', 'HYPOTHESIS'), help='Compute Word Error Rate (WER) between reference and hypothesis text files')
    args = parser.parse_args()

    if args.wer:
        with open(args.wer[0], 'r', encoding='utf-8') as f:
            reference = f.read()
        with open(args.wer[1], 'r', encoding='utf-8') as f:
            hypothesis = f.read()
        wer = compute_wer(reference, hypothesis)
        print(f"Word Error Rate (WER): {wer:.3f}")
        return

    config = AppConfig(realtime=args.realtime)
    pipeline = VideoToTextPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
