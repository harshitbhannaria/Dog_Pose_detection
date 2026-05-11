import cv2
import os
import numpy as np


IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'
OUT_DIR = 'data/output'
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["L_Eye", "R_Eye", "Nose"]

images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            # YOLO Parse
            xc, yc, bw, bh = map(float, parts[1:])
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # searching for the landmarks inside ROI
            roi = img[y1:y2, x1:x2] # HERE WE DEFINE THE ROI - Region of interest based on the parsed values from yolo format of the labels --> stored in data/labels directory.
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            rw, rh = gray_roi.shape[1], gray_roi.shape[0]

            def get_darkest_in_zone(z_x1, z_y1, z_x2, z_y2):
                zone = gray_roi[int(z_y1):int(z_y2), int(z_x1):int(z_x2)]
                if zone.size == 0: return (0, 0)
            # Find the absolute darkest pixel in this specific zone
                _, _, min_loc, _ = cv2.minMaxLoc(zone)
                return (min_loc[0] + z_x1 + x1, min_loc[1] + z_y1 + y1)
            # Define Zones (Relative to your manual box)
            # Eyes are in the top half, Nose in the bottom center
            pts = [
                get_darkest_in_zone(rw*0.1, rh*0.2, rw*0.5, rh*0.5), # L_Eye Zone
                get_darkest_in_zone(rw*0.5, rh*0.2, rw*0.9, rh*0.5), # R_Eye Zone
                get_darkest_in_zone(rw*0.3, rh*0.5, rw*0.7, rh*0.9)  # Nose Zone
            ]
            # --- DRAW RESULTS ---
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img,"", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for idx, pt in enumerate(pts):

                cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)
                cv2.putText(img, LABELS[idx], (int(pt[0])+10, int(pt[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    
    cv2.imwrite(os.path.join(OUT_DIR, f"OUTPUT_{img_name}"), img) #SO that the output is stored in data/output directory of dog_pose_project.

print(f"\nDone! Results in {OUT_DIR}")