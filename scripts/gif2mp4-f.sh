#!/bin/bash
font="/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
for i in *.gif; 
    do name=`echo "$i" | cut -d'.' -f1`
    # below is libx264
    # ffmpeg -i "$i" -movflags faststart -pix_fmt yuv420p -vf "scale=10*trunc(iw/2)*2:10*trunc(ih/2)*2:flags=neighbor, drawtext=fontfile=${font}: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=10*2: box=1: boxcolor=white: boxborderw=5, drawtext=fontfile=${font}: text='${name}': start_number=1: x=(w-tw)/2: y=lh: fontcolor=black: fontsize=10*2: box=1: boxcolor=white: boxborderw=5"  "${i%.*}.mp4"; 
    # libx264
    ffmpeg -i "$i" -movflags faststart -vcodec libx264 -pix_fmt yuv420p -vf "scale=10*trunc(iw/2)*2:10*trunc(ih/2)*2:flags=neighbor, drawtext=fontfile=${font}: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=10*2: box=1: boxcolor=white: boxborderw=5, drawtext=fontfile=${font}: text='${name}': start_number=1: x=(w-tw)/2: y=lh: fontcolor=black: fontsize=10*2: box=1: boxcolor=white: boxborderw=5"  "${i%.*}.mp4"; 
done