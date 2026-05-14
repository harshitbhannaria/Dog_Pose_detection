# Dog Pose Landmark Detection

This project detects 3 key landmarks (Nose, L_Eye, R_Eye) on a dog's face using manual bounding box labels and a pixel-intensity search grid.

## How it Works
1. The script reads manual face boundaries from the `data/labels/` folder.
2. It divides the boundary into search zones.
3. It identifies the darkest pixel clusters (intensity minima) within those zones to pinpoint the eyes and nose.

## Output mentioned in data/output directory 
