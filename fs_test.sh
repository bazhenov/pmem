#!/usr/bin/env bash

set -e

mount -t nfs -o nolocks,vers=3,tcp,port=11111,mountport=11111,soft 127.0.0.1:/ mnt/
trap 'echo Unmounting; umount mnt' EXIT

sleep 1
echo "Creating directory..."
mkdir mnt/dir

echo "Removing directory..."
rmdir mnt/dir

echo "Creating File..."
touch mnt/file

echo "Writing file..."
date > mnt/file

echo "Echoing file..."
cat mnt/file

echo "Deleting file..."
rm -f mnt/file
