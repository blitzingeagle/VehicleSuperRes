import argparse

parser = argparse.ArgumentParser(description="Vehicle Super Resolution Converter")
subparsers = parser.add_subparsers(help="commands")

image_parser = subparsers.add_parser("image", help="Convert Image")
image_parser.add_argument("-i", type=str, default="./images", metavar="INPUT", help="Input image batch directory (default: ./images)")
image_parser.add_argument("-o", type=str, default="./output", metavar="OUTPUT", help="Output image directory (default: ./output)")
image_parser.add_argument("-v", "--verbose", action="store_true")
image_parser.add_argument("-w", type=str, default="weights-beta.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
image_parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")
image_parser.set_defaults(which="image")

video_parser = subparsers.add_parser("video", help="Convert Video")
video_parser.add_argument("-i", type=str, default="./videos", metavar="INPUT", help="Input video directory (default: ./videos)")
video_parser.add_argument("-o", type=str, default="./frames", metavar="OUTPUT", help="Output image directory (default: ./frames)")
video_parser.add_argument("-v", "--verbose", action="store_true")
video_parser.add_argument("-w", type=str, default="weights-beta.pth", metavar="WEIGHTS", help="Path to weights file (default: weights-beta.pth)")
video_parser.add_argument("--ext", type=str, default="(keep)", metavar="ext", help="File extension for output (default: <uses the same extension as input>)")
video_parser.set_defaults(which="video")

args = parser.parse_args()

print(args.which)
