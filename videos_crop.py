# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This script is licensed under the MIT License.

import argparse
import multiprocessing as mp
from multiprocessing import get_context
import os
from functools import partial
from time import time as timer

import subprocess
import ffmpeg
from tqdm import tqdm
from subprocess import Popen, PIPE
from decimal import Decimal, DivisionByZero
import functools
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True,
                    help='Dir containing youtube clips.')
parser.add_argument('--clip_info_file', type=str, required=True,
                    help='File containing clip information.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Location to dump outputs.')
parser.add_argument('--num_workers', type=int, default=8,
                    help='How many multiprocessing workers?')
args = parser.parse_args()

@functools.lru_cache(maxsize=2048)
def get_h_w_fps(filepath):
    probe = ffmpeg.probe(filepath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    height = int(video_stream['height'])
    width = int(video_stream['width'])
    
    # Extract avg_frame_rate and convert to Decimal FPS
    avg_frame_rate = video_stream['avg_frame_rate']
    numerator, denominator = map(int, avg_frame_rate.split('/'))
    if denominator != 0:  # Prevent division by zero
        fps = Decimal(numerator) / Decimal(denominator)
    else:
        fps = Decimal(0)  # Handle division by zero, if applicable
    
    return height, width, fps

def frame_to_timestamp(frame_index: int, frame_rate: Decimal) -> str:
    # Calculate the time position in seconds
    time_position_seconds = Decimal(frame_index) / frame_rate
    
    # Convert the total seconds into hours, minutes, and seconds
    hours, remainder = divmod(time_position_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format the timestamp string
    timestamp = "{:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), float(seconds))
    
    return timestamp


def parse_clip_params(clip_params):
    video_name, H, W, S, E, L, T, R, B = clip_params.strip().split(',')
    H, W, S, E, L, T, R, B = int(H), int(W), int(S), int(E), int(L), int(T), int(R), int(B)
    return video_name, H, W, S, E, L, T, R, B

def trim_and_crop(input_dir, output_dir, clip_params):
    video_name, H, W, S, E, L, T, R, B = parse_clip_params(clip_params)
    output_filename = '{}_S{}_E{}_L{}_T{}_R{}_B{}.mp4'.format(video_name, S, E, L, T, R, B)
    output_filepath = os.path.join(output_dir, output_filename)
    if os.path.exists(output_filepath):
        print('Output file %s exists, skipping' % (output_filepath))
        return

    input_filepath = os.path.join(input_dir, video_name)

    for ext in ['.mp4', '.mkv', '.webm']:
        if os.path.exists(input_filepath + ext):
            input_filepath += ext
            break

    if not os.path.exists(input_filepath):
        print('Input file %s does not exist, skipping' % (input_filepath))
        return

    h, w, fps = get_h_w_fps(input_filepath)
    start_ts = frame_to_timestamp(S + 1, fps)
    end_ts = frame_to_timestamp(E, fps)

    if (E - S) / fps < 4:
        print('Clip under 4 seconds, skipping:', output_filename)
        return

    t = int(T / H * h)
    b = int(B / H * h)
    l = int(L / W * w)
    r = int(R / W * w)
    
    stream = ffmpeg.input(input_filepath, ss=start_ts, to=end_ts)
    video = stream.video
    audio = stream.audio
    video = ffmpeg.crop(video, l, t, r-l, b-t)
    stream = ffmpeg.output(
        audio,
        video,
        output_filepath,
        **{
            "c:v": "h264_nvenc",
            "preset": "slow",
            "b:v": "0",
            "cq:v": "24",
            "rc:v": "vbr",
        }
    )
    args = stream.get_args()
    command = ["ffmpeg", "-loglevel", "quiet"] + args
    return_code = subprocess.call(command)
    success = return_code == 0
    if not success:
        print('Command failed:', command)


if __name__ == '__main__':
    # Read list of videos.
    clip_info_by_video = defaultdict(list)
    with open(args.clip_info_file) as fin:
        for line in fin:
            video_name = parse_clip_params(line.strip())[0]
            clip_info_by_video[video_name].append(line.strip())
    
    # only include longest 4 clips (by E - S) for each video
    clip_info = []

    for video_name, clip_info_list in clip_info_by_video.items():
        def get_length_in_frames(clip_params):
            _, _, _, S, E, _, _, _, _ = parse_clip_params(clip_params)
            return E - S

        clip_info_list.sort(key=lambda x: get_length_in_frames(x), reverse=True)
        clip_info.extend(clip_info_list[:4])

    print('Total clips:', len(clip_info))

    # Create output folder.
    os.makedirs(args.output_dir, exist_ok=True)

    # Download videos.
    downloader = partial(trim_and_crop, args.input_dir, args.output_dir)

    start = timer()
    pool_size = args.num_workers
    print('Using pool size of %d' % (pool_size))
    with get_context("spawn").Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, clip_info), total=len(clip_info)))
    print('Elapsed time: %.2f' % (timer() - start))
