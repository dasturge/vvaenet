#!/usr/bin/env bash

while IFS= read -r FILEID; do
  CONFIRMATION=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=${CONFIRMATION}&id=$FILEID" -O data.tar
  rm -rf /tmp/cookies.txt
  tar xf data.tar
  rm -f data.tar
done < $(dirname ${BASH_SOURCE[0]})/drive_keys.txt

