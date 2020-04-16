#!/bin/bash
for i in *.MTS; do ffmpeg -i "$i" -c:v copy -c:a aac -strict experimental -b:a 128k "${i%.*}.mp4"; done