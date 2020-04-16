#!/bin/bash
font="/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
for i in *.gif; 
    do name=`echo "$i" | cut -d'.' -f1`
    ffmpeg -i "$i" -movflags faststart -pix_fmt yuv420p -vf "scale=100*trunc(iw/2)*2:100*trunc(ih/2)*2:flags=neighbor, drawtext=fontfile=${font}: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=100*2: box=1: boxcolor=white: boxborderw=5, drawtext=fontfile=${font}: text='${name}': start_number=1: x=(w-tw)/2: y=lh: fontcolor=black: fontsize=100*2: box=1: boxcolor=white: boxborderw=5"  "${i%.*}.mp4"; 
done