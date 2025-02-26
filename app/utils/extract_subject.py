#!/usr/bin/env python3
import sys
import json
from image_processing import ImageProcessor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
        
    image_path = sys.argv[1]
    processor = ImageProcessor()
    
    try:
        subject_path, mask = processor.extract_subject(image_path)
        # 保存掩码为图像
        mask_path = image_path.replace('.', '_mask.')
        import cv2
        import numpy as np
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        print(json.dumps({
            "subjectPath": subject_path,
            "maskPath": mask_path
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1) 