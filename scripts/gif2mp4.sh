for i in *.gif; do ffmpeg -i "$i"  -movflags faststart -pix_fmt yuv420p -vf "scale=100*trunc(iw/2)*2:100*trunc(ih/2)*2:flags=neighbor" "${i%.*}.mp4"; done

