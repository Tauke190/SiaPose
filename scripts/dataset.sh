#!/bin/bash
COCO_ROOT="${COCO_ROOT:-./datasets}"
mkdir -p "$COCO_ROOT"
cd "$COCO_ROOT"

# Download images
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip

# Download annotations
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract all
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

rm train2017.zip val2017.zip annotations_trainval2017.zip