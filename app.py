
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
    parser.add_argument('--cuda-device', type=int, default=0, help='CUDA GPU index to use (default: 0)')
    parser.add_argument('--wer', nargs=2, metavar=('REFERENCE', 'HYPOTHESIS'), help='Compute Word Error Rate (WER) between reference and hypothesis text files')
    parser.add_argument('--batch', action='store_true', help='Transcribe all audio/video files in the input directory')
    args = parser.parse_args()

    if args.wer:
        with open(args.wer[0], 'r', encoding='utf-8') as f:
            reference = f.read()
        with open(args.wer[1], 'r', encoding='utf-8') as f:
            hypothesis = f.read()
        wer = compute_wer(reference, hypothesis)
        print(f"Word Error Rate (WER): {wer:.3f}")
        return

    config = AppConfig(
        realtime=args.realtime,
        cuda_device=args.cuda_device
    )
    pipeline = VideoToTextPipeline(config)
    if args.batch:
        pipeline.run_batch()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
