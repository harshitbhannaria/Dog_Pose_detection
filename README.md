# Dog Pose Landmark Detection

This project detects 3 key landmarks (Nose, L_Eye, R_Eye) on a dog's face using manual bounding box labels and a pixel-intensity search grid.

## How it Works
1. The script reads manual face boundaries from the `data/labels/` folder.
2. It divides the boundary into search zones.
3. It identifies the darkest pixel clusters (intensity minima) within those zones to pinpoint the eyes and nose.

## Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/harshitbhannaria/Dog_Pose_detection.git](https://github.com/YOUR_USERNAME/dog_pose_project.git)
cd dog_pose_project