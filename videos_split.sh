#!/bin/bash

in_dir=$1
out_dir=$2

# Create the output directory if it doesn't exist
mkdir -p "$out_dir"

# Loop through all files in the input directory
for f in "$in_dir"/*; do
  if [[ -f $f ]]; then  # Check if it's a file
    filename=$(basename -- "$f")
    extension="${filename##*.}"
    basename="${filename%.*}"

    # Use ffmpeg to process the file, preserving the original extension
    ffmpeg -i "$f" -c copy -map 0 -segment_time 00:01:00 -f segment "$out_dir/${basename}_%04d.$extension"
  fi
done
