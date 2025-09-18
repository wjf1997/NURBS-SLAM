# NURBS-SLAM
NURBS-Based Continuous LiDAR-Visual-Inertial SLAM with Adaptive Keyframe Selection
In large-curvature motion, continuous-time SLAM
based on traditional B-splines often yields insufficiently robust
accuracy estimates. For this, this paper proposes a novel NURBSbased continuous LiDAR-visual-inertial SLAM with adaptive
keyframe selection. In front-end, we propose a keyframe selection
strategy specifically designed for large-curvature motion, which
incorporates overlap degree of 3D point cloud and 2D visual
correspondences based on frustum, while also establishing an
adaptive threshold for feature quadrant migration. In back-end,
trajectory is represented using the NURBS model. Based on distribution of keyframes, a non-uniform number of control points
is assigned to each segment, and weight of each control point
is subsequently encoded. The proposed algorithm is compared
with state-of-the-art algorithms and evaluated through ablation
experiments under large-curvature motion. The experimental
results demonstrate that the proposed algorithm effectively mitigates drift problem in large-curvature motion, and significantly
enhancing its robustness.<img width="1372" height="921" alt="image" src="https://github.com/user-attachments/assets/c59bfdcc-a404-4f67-af5e-9734ec9fada8" /><img width="1372" height="921" alt="image" src="https://github.com/user-attachments/assets/27a8f8fd-2fce-4250-bde2-aa201704ef7b" />
Our project is developed based on Coco-LIC. Therefore, the dependencies of our method are consistent with it, and there are no additional dependency requirements.
