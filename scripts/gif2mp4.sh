#!/bin/bash
for i in *.gif; do ffmpeg -i "$i" -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${i%.*}.mp4"; done

