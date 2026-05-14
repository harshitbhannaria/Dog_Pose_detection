import os
import cv2
import torch
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDHead
from torchvision.transforms import functional as F


IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'
OUT_DIR = 'data/output'
MODEL_PATH = 'dog_face_finetuned.pth'
os.makedirs(OUT_DIR, exist_ok=True)

def load_custom_ssd(num_classes=2):
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    # Use list comprehension to handle Sequential modules in torchvision
    in_channels = []
    for m in model.head.classification_head.module_list:
        for sub in m.modules():
            if hasattr(sub, 'in_channels'):
                in_channels.append(sub.in_channels)
                break
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels, num_anchors, num_classes)
    return model


def grid_search_landmarks(roi):
    #Performs a grid search within the ROI to find 3 specific landmarks.
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Define Grid Sectors (y_start, y_end, x_start, x_end)
    #Left Eye ->Top-Left quadrant
    l_eye_grid = gray[0:int(h*0.45), 0:int(w*0.5)]
    # Right Eye: Top-Right quadrant
    r_eye_grid = gray[0:int(h*0.45), int(w*0.5):w]
    #  Nose: Bottom-Center region
    nose_grid = gray[int(h*0.45):int(h*0.9), int(w*0.2):int(w*0.8)]

    def get_darkest_point(grid_patch):
        if grid_patch.size == 0: return (0, 0)
        _, _, min_loc, _ = cv2.minMaxLoc(grid_patch)
        return min_loc

    l_eye = get_darkest_point(l_eye_grid)
    r_eye = get_darkest_point(r_eye_grid)
    nose = get_darkest_point(nose_grid)

    return {
        "L_Eye": (l_eye[0], l_eye[1]),
        "R_Eye": (r_eye[0] + int(w*0.5), r_eye[1]),
        "Nose": (nose[0] + int(w*0.2), nose[1] + int(h*0.45))
    }

# TRAINING (Runs if .pth is missing) 
def train_if_needed():
    if os.path.exists(MODEL_PATH): return
    model = load_custom_ssd()
    for param in model.backbone.parameters(): param.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.0005)
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]

    for epoch in range(50):  # After 50 epochs goes for testing
        model.train()
        model.backbone.eval()
        for lbl in label_files:
            img_path = os.path.join(IMG_DIR, lbl.replace('.txt', '.png'))
            if not os.path.exists(img_path): img_path = os.path.join(IMG_DIR, lbl.replace('.txt', '.jpg'))
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (320, 320))
            with open(os.path.join(LABEL_DIR, lbl), 'r') as f:
                line = f.readline().split()
                if not line: continue
                cx, cy, nw, nh = map(float, line[1:])
                x1, y1, x2, y2 = (cx-nw/2)*320, (cy-nh/2)*320, (cx+nw/2)*320, (cy+nh/2)*320
            
            images = [F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
            targets = [{'boxes': torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32), 'labels': torch.tensor([1], dtype=torch.int64)}]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            sum(loss for loss in loss_dict.values()).backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)

# INFERENCE
def run_inference():
    model = load_custom_ssd()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    for img_name in os.listdir(IMG_DIR):
        if not img_name.lower().endswith(('.png', '.jpg')): continue
        img_orig = cv2.imread(os.path.join(IMG_DIR, img_name))
        h_o, w_o = img_orig.shape[:2]
        
        # SSD Input
        img_input = cv2.resize(img_orig, (320, 320))
        img_tensor = [F.to_tensor(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))]
        
        with torch.no_grad():
            preds = model(img_tensor)

        if len(preds[0]['boxes']) > 0:
            box = preds[0]['boxes'][0].cpu().numpy()
            x1, y1 = int(box[0] * w_o / 320), int(box[1] * h_o / 320)
            x2, y2 = int(box[2] * w_o / 320), int(box[3] * h_o / 320)

            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Grid Search within ROI
            roi = img_orig[max(0,y1):min(h_o,y2), max(0,x1):min(w_o,x2)]
            if roi.size > 0:
                landmarks = grid_search_landmarks(roi)
                # Left Eye (Blue), Right Eye (Blue), Nose (Red)
                cv2.circle(img_orig, (landmarks["L_Eye"][0] + x1, landmarks["L_Eye"][1] + y1), 5, (255, 0, 0), -1)
                cv2.circle(img_orig, (landmarks["R_Eye"][0] + x1, landmarks["R_Eye"][1] + y1), 5, (255, 0, 0), -1)
                cv2.circle(img_orig, (landmarks["Nose"][0] + x1, landmarks["Nose"][1] + y1), 5, (0, 0, 255), -1)

        cv2.imwrite(os.path.join(OUT_DIR, img_name), img_orig)

if __name__ == "__main__":
    train_if_needed()
    run_inference()
