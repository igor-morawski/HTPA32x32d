#!/bin/bash
#cut one video to many videos according to timestamps and the associated output file name
#add -c copy if you need speed (but some frames will be messed up), otherwise the command is as follows


echo "Please select the video you want to cut from the list"

files=$(ls *.mp4)
i=1

if test -z "$files"
then
    echo "No .mp4 files found in this directory. Aborting."
    exit 1
fi

for j in $files
do
echo "$i.$j"
file[i]=$j
i=$(( i + 1 ))
done


echo "Enter number (ENTER for 1)"
read input
if test -z "$input"
then
    input=1
fi
echo "You selected the file ${file[$input]}"

video=${file[$input]}

####################################################################

echo "Please select your .txt file with timestamps (format: each line is a tuple of "start end", e.g. "1:05, 1:27" or "start end name", e.g. "1:05 1:27 label")."

files=$(ls *.txt)
i=1

if test -z "$files"
then
    echo "No .txt files found in this directory. Aborting."
    exit 1
fi

for j in $files
do
echo "$i.$j"
file[i]=$j
i=$(( i + 1 ))
done


echo "Enter number (ENTER for 1)"
read input
if test -z "$input"
then
    input=1
fi
echo "You selected the file ${file[$input]}"

txt=${file[$input]}

idx=0
while IFS= read -r line || [ -n "$line" ];
    do
        set -- $line
        if test -z "$3"
        then
            output="output$idx.mp4"
            idx=$((idx+1))
        else
            output="$3.mp4"
        fi
        ffmpeg -n -t $2 -i $video -ss $1 $output </dev/null
done < $txt