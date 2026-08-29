# 📐 The Mathematical Foundations of GeoPrior & SSAKS (Explained Simply)

This document breaks down the mathematical concepts behind our automated drone 3D digital twin system into intuitive, visual explanations without unnecessary academic jargon.

---

## 🧭 Executive Summary: The 3 Core Mathematical Challenges

When a drone records a flight and we want to turn that video into a photorealistic, metric-accurate 3D digital twin, the computer must solve three fundamental math problems:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. THE DATA FLOOD PROBLEM (SSAKS Keyframe Selection)                   │
│    How do we filter 1,916 video frames down to 193 high-quality frames │
│    without losing any 3D geometric coverage?                           │
├────────────────────────────────────────────────────────────────────────┤
│ 2. THE CURVED EARTH PROBLEM (Geodesic Transforms)                      │
│    How do we translate GPS angles (Lat/Lon) into millimeter-accurate   │
│    metric Cartesian coordinates (East, North, Up meters)?              │
├────────────────────────────────────────────────────────────────────────┤
│ 3. THE 3D GAUSSIAN SPLATTING PROBLEM (Differentiable Rendering)        │
│    How do millions of 3D fuzzy "paint splats" project onto a 2D camera │
│    sensor and learn the true shape of buildings via calculus?          │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 🎥 1. SSAKS: How the Computer "Sees" Blur, Motion & Novelty

### 🛑 The Core Dilemma: The $O(N^2)$ Matching Explosion
If you take a 1-minute 4K video at 30 FPS, you get **1,916 individual photographs**. 
- To find matching 3D points between all pairs of photos, the computer must make $\frac{N(N-1)}{2} \approx \mathbf{1,834,570\text{ pairwise comparisons}}$.
- But if the drone was hovering in place for 10 seconds, 300 of those photos are 100% identical duplicates.
- **SSAKS** solves this by acting like a strict photo editor using three mathematical filters.

---

### A. Stage 1 Math: The Laplacian Blur Detector ($\nabla^2 I$)

#### 💡 The Intuition
Imagine walking across a photograph:
- On a **sharp edge** (like the crisp edge of a dark roof against a bright sky), your elevation jumps from $0$ to $255$ in a single pixel step. The slope (first derivative) is steep, and the acceleration (second derivative) is massive.
- On a **blurry edge**, the dark color slowly bleeds into the bright color over 20 pixels like a gentle ramp. The acceleration (second derivative) is nearly zero.

```
       SHARP EDGE (High Variance)                  BLURRY EDGE (Low Variance)
           Bright                                      Bright
             ┌───────                                      . - '
             │                                         . - '
             │                                     . - '
      ───────┘                              ─────── '
       Dark                                  Dark
     (Sudden Cliff => Var > 65)             (Smooth Ramp => Var < 65)
```

#### 🔢 The Math
The discrete Laplacian operator calculates the 2D second-order spatial derivative:
$$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

We compute the statistical **variance** ($\sigma^2$) across all pixels:
$$\text{Blur Score} = \text{Variance}(\nabla^2 I) = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu \right)^2$$

- If $\text{Blur Score} \ge 65.0 \implies$ **Sharp Frame (KEEP)**.
- If $\text{Blur Score} < 65.0 \implies$ **Vibration Blur (DROP)**.

---

### B. Stage 1 Math: Optical Flow Displacement ($\|\vec{u}\|$)

#### 💡 The Intuition
The computer tracks how many pixels objects shifted between consecutive frames.
- If things shifted **$< 0.15\text{ pixels}$**: The drone is hovering motionless. Taking another photo gives zero new 3D information $\implies$ **DROP**.
- If things shifted **$> 140\text{ pixels}$**: The drone or gimbal jerked violently, causing motion smearing $\implies$ **DROP**.
- If displacement is **between $0.15$ and $140\text{ pixels}$**: Smooth flight $\implies$ **PASS**.

#### 🔢 The Math
Dense displacement vector $\vec{u}(x, y) = [u_x, u_y]^T$:
$$\|\vec{u}\| = \sqrt{u_x^2 + u_y^2}$$

---

### C. Stage 2 Math: Visual Novelty & Cosine Similarity ($\cos \theta$)

#### 💡 The Intuition
Even if two photos are sharp, do they show a **new viewpoint**?
- The computer converts every photo into an **arrow (vector)** in high-dimensional space.
- If two photos show the exact same perspective, their arrows point in the exact same direction ($\theta \approx 0^\circ \implies \cos \theta \approx 1.0$).
- If the drone moved around a building to reveal a new facade, the arrow swings to a new angle.

```
                  Angle θ between two camera view vectors
                              Vector B (New Viewpoint)
                               ↗
                              / 
                             /  θ (Angle > 28°)
                            / 
                           └───────► Vector A (Previous Keyframe)
                       cos(θ) < 0.88 => NOVEL VIEW (KEEP!)
```

#### 🔢 The Math
Given unit-normalized feature vectors $\mathbf{a}$ and $\mathbf{b}$ ($\|\mathbf{a}\| = 1, \|\mathbf{b}\| = 1$):
$$\text{Similarity} = \cos(\theta) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^D a_i b_i$$

- If $\cos(\theta) < 0.88 \implies$ **New Perspective (KEEP KEYFRAME)**.
- If $\cos(\theta) \ge 0.88 \implies$ **Redundant Viewpoint (DROP)**.

---

# 🌍 2. Geodesic Math: From Curved Earth to Flat 3D Meters

### 🛑 The Problem: GPS is Angular, 3D Rendering is Linear
- GPS gives coordinates as angles on an ellipsoid: $\text{Latitude } \phi = 43.0428^\circ\text{ N}, \text{ Longitude } \lambda = -77.6663^\circ\text{ W}$.
- But 3D rendering engines (and 3D Gaussians) require flat metric Cartesian coordinates in meters: $(X, Y, Z)$.

```
   WGS84 Curved Ellipsoid                Local Flat Tangent Plane (ENU)
        (GPS Degrees)                             (Metric Meters)
             ╭───╮                                       ▲ +Z (Up / Height)
           ╭─╯   ╰─╮                                     │
          │  ● (GPS)│         ════════════════►         │  ● Drone (12m E, 45m N, 20m Up)
           ╰─╮   ╭─╯      Geodesic Transformations       │ /
             ╰───╯                                       └───────► +X (East)
                                                        /
                                                       ▼ +Y (North)
```

---

### 🔢 The 2-Step Mathematical Translation

#### Step 1: Geodetic $\to$ Earth-Centered Earth-Fixed (ECEF)
We model Earth as an oblate spheroid with equatorial radius $a = 6,378,137\text{ m}$ and flattening $f = 1/298.257223563$.

The curvature radius $N(\phi)$ at latitude $\phi$ is:
$$N(\phi) = \frac{a}{\sqrt{1 - (2f - f^2) \sin^2 \phi}}$$

The 3D coordinate from the center of the Earth is:
$$X = (N + h) \cos \phi \cos \lambda, \quad Y = (N + h) \cos \phi \sin \lambda, \quad Z = (N(1 - e^2) + h) \sin \phi$$

#### Step 2: ECEF $\to$ Local Tangent East-North-Up (ENU)
We place a flat coordinate grid right at the drone's takeoff point $(X_0, Y_0, Z_0)$:
$$\begin{bmatrix} x_{\text{East}} \\ y_{\text{North}} \\ z_{\text{Up}} \end{bmatrix} = \begin{bmatrix} -\sin \lambda & \cos \lambda & 0 \\ -\sin \phi \cos \lambda & -\sin \phi \sin \lambda & \cos \phi \\ \cos \phi \cos \lambda & \cos \phi \sin \lambda & \sin \phi \end{bmatrix} \begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$

**Result**: $+X$ is meters East, $+Y$ is meters North, $+Z$ is altitude above ground in meters.

---

# 🔮 3. 3D Gaussian Splatting: How Math Renders Reality

### 💡 The Intuition
Instead of building a 3D model out of flat rigid triangles (polygons), 3D Gaussian Splatting fills the world with **millions of tiny, semi-transparent, fuzzy 3D "paint splats"** (Gaussians).

Each Gaussian has:
1. A **Position in 3D space**: $\mu = (x, y, z)$.
2. A **3D Shape & Size**: Stretched along 3 axes via scale vector $\mathbf{s} = (s_x, s_y, s_z)$.
3. A **3D Rotation**: Rotated in space via a quaternion $\mathbf{q} = (w, x, y, z)$.
4. An **Opacity**: How solid or see-through it is ($\alpha \in [0, 1]$).
5. A **Color**: Stored as directional light coefficients (Spherical Harmonics).

```
   1. 3D Gaussian in Space         2. Camera Projection           3. 2D Screen Splat
            (World)                      (Perspective)                 (Pixels)
             .---.                                                       .---.
           /   |   \               ┌─────────────┐                     /   |   \
          | ---+--- |    ═══════►  │ Camera Lens │    ═══════►        | ---+--- |
           \   |   /               └─────────────┘                     \   |   /
             '---'                                                       '---'
         Covariance Σ                   Jacobian J                 2D Covariance Σ_2D
```

---

### 🔢 The Differentiable Projection Formula

To draw a 3D Gaussian onto a 2D computer screen from camera viewpoint $\mathbf{W} = [\mathbf{R} | \mathbf{t}]$, its 3D covariance $\Sigma$ is projected onto the 2D pixel grid using the affine transformation (Zwicker / Kerbl formulation):
$$\Sigma_{\text{2D}} = \mathbf{J} \mathbf{W} \Sigma \mathbf{W}^T \mathbf{J}^T$$

Where $\mathbf{J}$ is the **Jacobian matrix** of the camera's perspective projection lens:
$$\mathbf{J} = \begin{bmatrix} \frac{f_x}{z} & 0 & -\frac{f_x \cdot x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y \cdot y}{z^2} \end{bmatrix}$$

---

### 🎨 Volumetric Alpha Blending (Drawing the Pixels)
To calculate the color $C$ of any pixel on your screen, we sort all the overlapping splats from front to back and blend them:
$$C = \sum_{i=1}^N \text{Color}_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1} (1 - \alpha_j)$$

$$\text{Transmittance } T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) \implies \text{"How much light can pass through all previous splats?"}$$

---

### 🎓 How the Model Learns: Calculus Backpropagation ($\nabla \mathcal{L}$)
1. The computer renders an image from the 3D Gaussians.
2. It compares the rendered image with the **real drone photo** using a composite loss function:
   $$\mathcal{L} = 0.8 \cdot \mathcal{L}_1(\text{Pixel Difference}) + 0.2 \cdot (1 - \text{SSIM}(\text{Structure Correlation}))$$
3. Because every single equation above is differentiable (smooth curves, no discrete jumps), calculus tells the optimizer the exact partial derivatives:
   $$\frac{\partial \mathcal{L}}{\partial \mu} \text{ (Where to move it)}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{s}} \text{ (How to resize it)}, \quad \frac{\partial \mathcal{L}}{\partial \text{Color}} \text{ (What color to paint it)}$$

Over several hundred iterations, the fuzzy splats lock into place, forming razor-sharp roofs, crisp windows, and detailed terrain!

---

# 📊 Summary Reference Sheet

| Concept | The Math Equation | Simple English Translation |
| :--- | :--- | :--- |
| **Blur Detection** | $\text{Var}(\nabla^2 I) \ge 65.0$ | Are there sharp pixel cliffs or gentle blurry ramps? |
| **Optical Flow** | $0.15 \le \|\vec{u}\| \le 140.0$ | Did things move enough to avoid duplicates, but not so fast they smeared? |
| **Visual Novelty** | $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} < 0.88$ | Does this frame show a new camera angle compared to previous keyframes? |
| **Coordinate Datum** | $\mathbf{P}_{\text{ENU}} = \mathbf{R} (\mathbf{P}_{\text{ECEF}} - \mathbf{P}_0)$ | Lay a flat metric sheet of paper tangent to where the drone took off. |
| **2D Projection** | $\Sigma_{\text{2D}} = \mathbf{J} \mathbf{W} \Sigma \mathbf{W}^T \mathbf{J}^T$ | Flatten a 3D fuzzy football onto the 2D camera sensor plane. |
| **Alpha Blending** | $C = \sum c_i \alpha_i \prod (1 - \alpha_j)$ | Stack semi-transparent colored glass layers from front to back. |
