# FSTINet: Spatial-Temporal Integrated Network for Class-Agnostic Motion Prediction with Frequency-Spatial Fusion
This repository is the official implementation of **FSTINet**, a novel framework designed to enhance class-agnostic motion prediction accuracy and robustness in autonomous driving systems. FSTINet addresses key challenges in spatio-temporal dependency modeling and dynamic feature capture by integrating spatial-temporal dynamics with frequency-domain information, as proposed in the paper:  
*FSTINet: Spatial-Temporal Integrated Network for Class-Agnostic Motion Prediction with Frequency-Spatial Fusion*  
*(Paper File: FSTINet___Yuzhuo_Feng___2025 (1).pdf | Code will be released soon)*




## Overview
Accurate motion prediction of traffic participants is a critical prerequisite for safe and efficient autonomous driving. Existing methods often face two core limitations:
1. **Insufficient spatio-temporal dependency modeling**: Conventional strategies (e.g., 3D max-pooling, global aggregation) easily lose long-term temporal dynamics and fine-grained local variations, leading to poor adaptability in high-speed scenarios {insert\_element\_0\_}.
2. **Limited dynamic feature capture**: Effective prediction requires balancing low-frequency global context and high-frequency local details, but existing approaches struggle to fuse these two types of features, often confusing subtle motion changes with background noise {insert\_element\_1\_}.

FSTINet resolves these issues by introducing two core modules that unify spatial, temporal, and frequency-domain information, achieving robust performance in complex dynamic environments {insert\_element\_2\_}.


## Key Contributions
The main contributions of FSTINet, as detailed in *FSTINet___Yuzhuo_Feng___2025 (1).pdf*, are:
1. **Hierarchical Temporal-Spatial Integrated Module (HTSIM)**: Integrates temporal max-pooling with a hierarchical interaction learning mechanism to establish bidirectional cross-scale correlations. This design captures both long-term temporal dependencies and fine-grained dynamic changes in traffic scenarios {insert\_element\_3\_}.
2. **Frequency-Spatial Fusion Module (FSFM)**: Leverages 2D Discrete Wavelet Transform (2D-DWT) to decompose features into multiple frequency bands. It applies differentiated fusion strategies for low-frequency (global semantics) and high-frequency (local details) components, enhancing feature diversity and dynamic representation capability {insert\_element\_4\_}.
3. **State-of-the-art performance on nuScenes**: Extensive experiments show FSTINet outperforms existing methods, especially in high-speed scenarios (speed > 5 m/s), while maintaining real-time inference capability (16.6 ms per frame on a single NVIDIA RTX 4090 GPU) {insert\_element\_5\_}.


## Method Architecture
### 1. Overall Pipeline
FSTINet adopts an encoder-decoder structure with four key components, as illustrated in the paper’s Fig. 2:
- **Encoder**: A ResNet34-based backbone (pretrained on ImageNet-1K) extracts multi-scale Bird’s Eye View (BEV) features from LiDAR point cloud sequences. 3D convolutional layers are inserted between ResBlocks to explicitly refine temporal dependencies {insert\_element\_6\_}.
- **HTSIM**: Embedded in skip connections between the encoder and decoder, HTSIM refines skip-connection features by suppressing temporal redundancy and enhancing cross-scale spatio-temporal interaction {insert\_element\_7\_}.
- **FSFM**: Positioned at the network bottleneck, FSFM fuses frequency-domain cues (from wavelet decomposition) with spatial features to generate compact, discriminative representations {insert\_element\_8\_}.
- **Decoder**: Follows the structure of MotionNet [11], progressively upsampling fused features and integrating them with HTSIM-refined skip features to output motion states, category distributions, and displacement fields {insert\_element\_9\_}.

### 2. Core Modules
#### (a) HTSIM: Hierarchical Temporal-Spatial Integrated Module
HTSIM’s workflow (paper Fig. 3) includes two key steps:
- **Temporal Max-Pooling (TMP)**: Aggregates features along the temporal axis to suppress redundancy, preserving salient motion cues while filtering short-term fluctuations {insert\_element\_10\_}.
- **Hierarchical Interaction Learning Mechanism (HILM)**: Divides features into channel groups, applies unified transformations (SS2D for long-range context, WTConv for frequency-aware filtering, ECA for channel attention), and establishes bidirectional cross-scale correlations via group-wise matrix multiplication. This selectively enhances stable spatio-temporal patterns {insert\_element\_11\_}.

#### (b) FSFM: Frequency-Spatial Fusion Module
FSFM’s core is **Wavelet-Based Multi-Frequency Fusion (WMFF)**, as shown in the paper’s Fig. 4:
- **Wavelet Decomposition**: Uses Haar wavelet basis to split input features into low-frequency (LL, global contours/background) and high-frequency (LH/HL/HH, edges/fast motion) components {insert\_element\_12\_}.
- **Differentiated Fusion**: Low-frequency components are concatenated to preserve global semantics; high-frequency components are fused element-wise to emphasize transient signals and suppress noise. The fused features are further refined with ECA and residual connections {insert\_element\_13\_}.


## Dataset Preparation (For Future Use)
When code is released, users will need to prepare the nuScenes dataset (the benchmark used in FSTINet’s experiments) as follows:

### nuScenes Dataset
- **Source**: Download the large-scale nuScenes dataset (v1.0) from the [official website](https://www.nuscenes.org/nuscenes#download), which includes multimodal data (LiDAR point clouds, images, annotations) and high-precision metadata {insert\_element\_14\_}.
- **Data Split**: Follow the standard split from MotionNet [11]: 500 training scenes, 100 validation scenes, and 250 test scenes {insert\_element\_15\_}.

### Data Preprocessing Pipeline
The preprocessing steps (to be implemented in future code) will include:
1. **Spatial Clipping**: Restrict LiDAR point clouds to the region [-32 m, 32 m] × [-32 m, 32 m] × [-3 m, 2 m] (critical area around the ego-vehicle) {insert\_element\_16\_}.
2. **Voxelization**: Quantize point clouds into a regular grid with voxel dimensions 0.25 m × 0.25 m × 0.4 m {insert\_element\_17\_}.
3. **Ego-Motion Compensation**: Align multi-frame point clouds (current frame + 4 previous frames) to the current coordinate system to capture continuous temporal changes {insert\_element\_18\_}.
4. **BEV Feature Generation**: Convert 3D voxel grids into 2D BEV pseudo-images (input tensor shape: 5 × 13 × 256 × 256, where 5 = number of frames, 13 = feature channels) {insert\_element\_19\_}.


## Training & Inference (For Future Use)
Detailed training and inference instructions will be provided when code is released. Key parameters (as specified in *FSTINet___Yuzhuo_Feng___2025 (1).pdf*) include:
- **Training Objective**: Multi-task loss combining cross-entropy loss (cell classification, motion state estimation) and Smooth L1 loss (motion prediction), with balancing weights λ_cls, λ_motion, λ_state {insert\_element\_20\_}.
- **Optimization**: Adam optimizer, initial learning rate 0.0016 (decayed by 0.5 every 10 epochs), batch size 8, 30 total epochs {insert\_element\_21\_}.
- **Inference**: Predicts displacement changes over the next 1 second, with runtime controlled within 100 ms to meet autonomous driving real-time requirements {insert\_element\_22\_}.


## Experimental Results
All results are derived from experiments on the nuScenes dataset, as reported in *FSTINet___Yuzhuo_Feng___2025 (1).pdf*.

### 1. Comparison with State-of-the-Art Methods
FSTINet outperforms existing class-agnostic motion prediction methods, particularly in dynamic scenarios:

| Method          | Static (Mean Error, m) | Low-Speed (≤5 m/s, Mean Error, m) | High-Speed (>5 m/s, Mean Error, m) |
|-----------------|-------------------------|-----------------------------------|------------------------------------|
| MotionNet [11]  | 0.0233                  | 0.2535                            | 1.0345                             |
| BE-STI [20]     | 0.0244                  | 0.2375                            | 0.9078                             |
| **FSTINet (Ours)** | 0.0267              | 0.2277                            | 0.8359                             |

*Note: Lower values indicate better performance. FSTINet reduces high-speed error from 1.0345 m (MotionNet) to 0.8359 m, a significant improvement {insert\_element\_23\_}.*

### 2. Ablation Studies
Ablation experiments confirm the effectiveness of each core module (paper Table III):

| Method          | HTSIM | FSFM | High-Speed Error (m) | Mean Category Accuracy (MCA, %) |
|-----------------|-------|------|----------------------|----------------------------------|
| Baseline (MotionNet) | × | × | 1.0345 | 71.3 |
| + ResBlock      | × | × | 0.9218 | 69.8 |
| + HTSIM         | √ | × | 0.8644 | 73.6 |
| + FSFM          | × | √ | 0.8541 | 73.6 |
| **FSTINet**     | √ | √ | 0.8359 | 75.9 |

*The combination of HTSIM and FSFM achieves the best performance, validating their synergy in fusing spatio-temporal and frequency features {insert\_element\_24\_}.*

### 3. Qualitative Analysis
As shown in the paper’s Fig. 5, FSTINet generates more accurate motion trajectories (represented by arrows) for fast-moving objects compared to MotionNet. It also improves cell classification (color-coded: gray = background, pink = vehicle, black = pedestrian, yellow = bike, red = others) by better distinguishing small or fast-moving targets {insert\_element\_25\_}.


