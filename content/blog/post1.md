+++
title = '[Review] Decoupling Human and Camera Motion from Videos in the Wild'
date = 2024-04-14T16:26:27+09:00
draft = false
+++

[Slahmr](https://github.com/vye16/slahmr) is a method that outputs multiple 3d human pose tracklets with correspending camera pose trajectories in a world coordinate frame, given a single monocular rgb camera.

This blog post aims to review 4 main components of this paper.

## Background

### Camera Projection Matrix

$$
s \begin{bmatrix} u \\\\ v \\\\ 1 \end{bmatrix} = \begin{bmatrix}
   f_x & 0 & c_x\\\\
   0 & f_y & c_y \\\\ 
   0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_{1} \\\\
r_{21} & r_{22} & r_{23} & t_{2} \\\\
r_{31} & r_{32} & r_{33} & t_{3} \\\\
\end{bmatrix} \begin{bmatrix}
X \\\\ Y \\\\ Z \\\\ 1
\end{bmatrix}
$$

The camera projection matrix is used to convert 3d points in the world to 2d uv coordinates in the image captured by camera.

$s$ : scale factor

$f_x, f_y$ : focal length

$c_x, c_y$  : camera center

$r_{ij}$ : rotation parameters

$t_i$ : translation parameters

### SLAM

SLAM stands for Simultaneous Localization and Mapping. This technique is used in computer vision and robotics to create map of the world along with the robot’s (i.e. camera) location trajectories in real time.

Many sensors are used in SLAM such as rgb camera, depth sensors, IMUs or LiDAR. In recent computer vision papers that adapts SLAM algorithm as a tool to retrieve camera pose trajectires, researches use state-of-the-art rgb deep slam model such as ‘DROID-SLAM’

## SLAHMR

![Untitled](/slahmrimages/Untitled.png)

### Data Preparation

1. Monocular RGB Input Frames
    
    Input to the SLAHMR pipeline is a consecutive rgb frames captured by moving camera which contains target human and background. Extract all frames of the video to prepare for SLAM and 3d human pose estimation & tracking.
    
2. Deep Visual-SLAM
    
    Run SFM to estimate relative camera motion between frames. This paper uses [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), the state-of-the-art deep visual slam system. 
    
    → world-to-camera transforms at time t: $\{\hat{R_t}, \hat{T_t}\}$
    
3. 3d human pose estimation & tracking
    
    Run [PHALP](https://github.com/brjathu/PHALP) to obtain initial human pose in the shape of [SMPL-H](https://smpl.is.tue.mpg.de/) parameter as well as tracklets of each human ids in the video.
    
    → SMPL-H pose parameters in camera coordinate frame $^cP^i_t$ = $\{^c\Phi^i_t, \Theta^i_t, \beta^i_t, ^c\Gamma^i_t\}$
    
    → $^c\Phi^i_t \in R^3$ : global orientation in camera coordinate frame
    
    →  $\Theta^i_t \in R^{22 \times 3}$ : body pose 
    
    → $\beta^i_t \in R^{16}$ : body shape parameter
    
    → $^c\Gamma^i_t \in R^{3}$ : root translation in camera coordinate frame
    
4. 2d joints coordniate
    
    Run [VITPose](https://github.com/ViTAE-Transformer/ViTPose) in each frame to obtain 2d joints coordinates to use for optimization
    
    →$x^i_t$ : 2d keypoints estimated by VITPose
    
    → $\psi^i_t$ : confidences
    

### Initializing people in the world

Pose parameter in camera coordinate frame  $^cP^i_t$ needs to be pushed into the world coordinate frame. The equation of doing this uses world to camera transformation estimated by SLAM.

$^w\phi^i_t = R^{-1}_t {}^c\phi^i_t$, $^w\Gamma^i_t = R^{-1}_t{}^c\Gamma^i_t - \alpha R^{-1}_t T_t$  → $^wP^i_t = \{^w\Phi^i_t, \Theta^i_t, \beta^i_t, ^w\Gamma^i_t\}$

with the estimated pose parameters, SMPL-H model outputs human joints coordinate in world coordinate frame

$$
^wJ^i_t = M(^w\Phi^i_t , \Theta^i_t, \beta^i) + ^w\Gamma^i_t
$$

### Joint re-projection loss (1st optimization stage)

$$
E_{data} = \sum^N_{i=1}\sum^T_{t=1}\psi^i_t\rho(\Pi_K(R_t \cdot ^wJ^i_t + \alpha T_t) -x^i_t
$$

- $\Pi_K(R_t \cdot ^wJ^i_t + \alpha T_t)$  : the extrinsics $R_t , T_t$ apply the camera movement to the world joint coordinate and the function $\Pi_k$ applies intrinsic $K$ to the world joints to project them to the camera cooridnate frame
- Only optimize global orientation and root translation $\{ ^w\Phi^i_t , ^w\Gamma^i_t \}$ in the first stage since the loss is very under constrained.

$$
\underset{\{\{^w\Phi^i_t, ^w\Gamma^i_t\}^T_{t=1}\}^N_{i=1}}{min} \lambda_{data}E_{data}
$$

### Smoothing trajectories in the world (2nd optimization stage)

- joint smoothness loss

$$
E_{smooth} = \sum^N_{i=1}\sum^T_{t=1} || J^i_t - J^i_{t+1}||^2
$$

- shape & pose loss
    
    $$
    E_{\beta} = \sum^N_{i=1}||\Beta^i||^2 , ~ E_{pose} = \sum^N_{i=1}\sum^T_{t=1}||\zeta^i_t||^2 ~(\zeta^i_t \in R^{32})
    $$
    
    $\zeta$ is latent of [VPoser](https://github.com/nghorbani/human_body_prior) model that encodes pose parameters ($\Theta$) into latent representation
    
- optimize for camera scale $\alpha$ and smpl pose parameters ($^wP^i_t$)

$$
\underset{\alpha , \{\{^wP^i_t\}^T_{t=1}\}^N_{i=1}}{min} \lambda_{data}E_{data}+\lambda_{\beta}E_{\beta} + \lambda_{pose}E_{pose}+\lambda_{smooth}E_{smooth}
$$

### Incorporating learned human motion priors (3rd optimization stage)

[HuMoR](https://github.com/davrempe/humor) model is incorporated into the 3rd optimization stage to prevent irregular human motions. HuMoR takes as input noisy human pose estimations and generates plausible human motions.

![HuMoR](/slahmrimages/HuMoR__3D_Human_Motion_Model_for_Robust_Pose_Estimation_(ICCV_2021)_6-35_screenshot.png)

Following HuMoR optimization method, Slahmr optimizaes for camera scale $\alpha$,  ground plane $g$, initial human trajectories $\{ s^0_0, ..., s^N_0\}$, transition latents $z^i_t$. HuMoR method is handled with details in another blog post. 

$$
\underset{\alpha, g, \{s^i_0\}^N_{i=0},\{\{z^i_t\}^T_{t=1}\}^N_{i=1}}{min} \lambda_{data}E_{data}+\lambda_{\beta}E_{\beta}+\lambda_{pose}E_{pose}+E_{prior}+E_{env}
$$

## Demo

Let’s take a look at the result of each optimization stage

- Initialization
    
    Here is raw output of PHALP estimation pushed back into the world coordinate frame with camera parameters estimated with DROID-SLAM
    
    ![ezgif-2-21259159c6.gif](/slahmrimages/ezgif-2-21259159c6.gif)
    
- 1st optimization stage
    
    each person’s global orientation and root translation is optimized in this stage
    
    ![ezgif-2-411821ccb6.gif](/slahmrimages/ezgif-2-411821ccb6.gif)
    
- 2nd optimization stage
    
    human pose parameters and camera scale are optimized in this stage
    
    ![ezgif-2-33117100b7.gif](/slahmrimages/ezgif-2-33117100b7.gif)
    
- 3rd optimization stage
    
    Human motion model is incorporated to refine the motions
    
    ![ezgif-2-82d8d28b38.gif](/slahmrimages/ezgif-2-82d8d28b38.gif)
    
- Final optimization result in grid
    
    ![ezgif-5-7b5677186d.gif](/slahmrimages/ezgif-5-7b5677186d.gif)