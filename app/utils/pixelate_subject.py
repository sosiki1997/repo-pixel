#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
from image_processing import ImageProcessor

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)
        
    subject_path = sys.argv[1]
    mask_path = sys.argv[2]
    pixel_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0  # 二值化
    
    processor = ImageProcessor()
    
    try:
        result_path = processor.pixelate_subject(subject_path, mask, pixel_size)
        print(json.dumps({
            "resultPath": result_path
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1) 