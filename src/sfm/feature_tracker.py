"""
Multi-View Feature Tracking & Correspondence Graph Module
==========================================================
Extracts robust 2D keypoints, performs epipolar-constrained reciprocal matching,
and chains correspondences into multi-view feature tracks for geometric triangulation.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class KeypointFeature:
    """Individual detected 2D keypoint."""
    x: float
    y: float
    response: float
    size: float
    octave: int = 0


@dataclass
class ImageFeatures:
    """Feature keypoints and descriptors extracted from a single keyframe."""
    image_idx: int
    image_name: str
    width: int
    height: int
    keypoints: np.ndarray        # (N, 2) float32 coordinates [x, y]
    descriptors: np.ndarray      # (N, D) float32 / uint8 descriptors
    colors: np.ndarray           # (N, 3) uint8 RGB colors sampled at keypoints


@dataclass
class PairwiseMatch:
    """Geometric match result between two keyframes."""
    src_idx: int
    dst_idx: int
    matches_src: np.ndarray      # (M, 2) inlier points in src image
    matches_dst: np.ndarray      # (M, 2) inlier points in dst image
    inlier_indices_src: np.ndarray # (M,) indices into src features
    inlier_indices_dst: np.ndarray # (M,) indices into dst features
    inlier_ratio: float
    essential_matrix: Optional[np.ndarray] = None
    relative_R: Optional[np.ndarray] = None
    relative_t: Optional[np.ndarray] = None


@dataclass
class FeatureTrack:
    """A 3D physical point observed across multiple camera views."""
    track_id: int
    observations: Dict[int, Tuple[float, float]] = field(default_factory=dict) # image_idx -> (x, y)
    color: np.ndarray = field(default_factory=lambda: np.array([128, 128, 128], dtype=np.uint8))
    point_3d: Optional[np.ndarray] = None
    reprojection_error: float = 0.0

    @property
    def track_length(self) -> int:
        return len(self.observations)


class FeatureTracker:
    """Extracts 2D features, matches view pairs, and builds multi-view feature tracks."""

    def __init__(
        self,
        max_features: int = 2000,
        ratio_thresh: float = 0.78,
        ransac_thresh_px: float = 2.0,
        min_inliers: int = 15,
    ):
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh_px = ransac_thresh_px
        self.min_inliers = min_inliers

        # Initialize detector: SIFT if available, otherwise ORB
        try:
            self.detector = cv2.SIFT_create(nfeatures=self.max_features, contrastThreshold=0.03)
            self.is_sift = True
        except Exception:
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.is_sift = False

    def extract_features(self, image: np.ndarray, image_idx: int, image_name: str = "") -> ImageFeatures:
        """Extract multi-scale keypoints, descriptors, and RGB colors."""
        if len(image.shape) == 3:
            h, w, _ = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image
        else:
            h, w = image.shape
            gray = image

        kps, descs = self.detector.detectAndCompute(gray, None)
        if kps is None or len(kps) == 0 or descs is None:
            return ImageFeatures(
                image_idx=image_idx,
                image_name=image_name,
                width=w,
                height=h,
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128 if self.is_sift else 32), dtype=np.float32),
                colors=np.empty((0, 3), dtype=np.uint8),
            )

        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        if not self.is_sift and descs.dtype != np.float32:
            descs = descs.astype(np.float32)

        # Sample RGB colors at keypoint locations
        colors = np.zeros((len(pts), 3), dtype=np.uint8)
        if len(image.shape) == 3:
            ix = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            colors = image[iy, ix, :]
            if colors.shape[1] == 3:
                # Convert BGR to RGB
                colors = colors[:, ::-1]

        return ImageFeatures(
            image_idx=image_idx,
            image_name=image_name,
            width=w,
            height=h,
            keypoints=pts,
            descriptors=descs,
            colors=colors,
        )

    def compensate_rolling_shutter(
        self,
        features: ImageFeatures,
        fx: float,
        fy: float,
        yaw_rate_rad_s: float,
        pitch_rate_rad_s: float = 0.0,
        readout_time_s: float = 0.0145 # DJI Mini 3 Pro 4K CMOS readout
    ) -> ImageFeatures:
        """
        Compensates CMOS rolling shutter scanline time-skew per Chapter 13.
        Maps keypoints to equivalent global-shutter center-frame exposure time.
        """
        if len(features.keypoints) == 0 or (abs(yaw_rate_rad_s) < 1e-4 and abs(pitch_rate_rad_s) < 1e-4):
            return features

        h = float(features.height)
        pts = features.keypoints.copy()
        
        # Relative scanline time offset tau in [-t_r/2, +t_r/2]
        tau = (pts[:, 1] / h - 0.5) * readout_time_s
        
        # Clamp angular velocities to physically plausible drone limits (< 90 deg/s = 1.57 rad/s)
        w_yaw = float(np.clip(yaw_rate_rad_s, -1.57, 1.57))
        w_pitch = float(np.clip(pitch_rate_rad_s, -1.57, 1.57))

        # Optical scanline correction: + du restores the center-frame projection
        du = fx * w_yaw * tau
        dv = fy * w_pitch * tau
        
        pts[:, 0] = pts[:, 0] + du
        pts[:, 1] = pts[:, 1] + dv

        return ImageFeatures(
            image_idx=features.image_idx,
            image_name=features.image_name,
            width=features.width,
            height=features.height,
            keypoints=pts,
            descriptors=features.descriptors,
            colors=features.colors
        )

    def match_pair(
        self,
        feat1: ImageFeatures,
        feat2: ImageFeatures,
        K: Optional[np.ndarray] = None
    ) -> Optional[PairwiseMatch]:
        """Reciprocal Lowe-ratio feature matching with Epipolar RANSAC gating."""
        if len(feat1.keypoints) < self.min_inliers or len(feat2.keypoints) < self.min_inliers:
            return None

        # FLANN or BF matcher
        if self.is_sift:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            knn_matches = matcher.knnMatch(feat1.descriptors, feat2.descriptors, k=2)
        except Exception:
            return None

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_inliers:
            return None

        src_pts = np.array([feat1.keypoints[m.queryIdx] for m in good_matches], dtype=np.float32)
        dst_pts = np.array([feat2.keypoints[m.trainIdx] for m in good_matches], dtype=np.float32)
        src_indices = np.array([m.queryIdx for m in good_matches], dtype=np.int32)
        dst_indices = np.array([m.trainIdx for m in good_matches], dtype=np.int32)

        # RANSAC Epipolar Filtering (Fundamental or Essential matrix)
        if K is not None:
            E, inlier_mask = cv2.findEssentialMat(
                src_pts, dst_pts, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.ransac_thresh_px
            )
            if E is None or inlier_mask is None:
                return None
            inlier_mask = inlier_mask.ravel().astype(bool)
            if np.sum(inlier_mask) < self.min_inliers:
                return None

            # Recover relative pose
            _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, src_pts[inlier_mask], dst_pts[inlier_mask], K)
        else:
            F, inlier_mask = cv2.findFundamentalMat(
                src_pts, dst_pts,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_thresh_px,
                confidence=0.999
            )
            if F is None or inlier_mask is None:
                return None
            inlier_mask = inlier_mask.ravel().astype(bool)
            E, R_rel, t_rel = None, None, None

        if np.sum(inlier_mask) < self.min_inliers:
            return None

        inlier_ratio = float(np.sum(inlier_mask) / max(1, len(good_matches)))

        return PairwiseMatch(
            src_idx=feat1.image_idx,
            dst_idx=feat2.image_idx,
            matches_src=src_pts[inlier_mask],
            matches_dst=dst_pts[inlier_mask],
            inlier_indices_src=src_indices[inlier_mask],
            inlier_indices_dst=dst_indices[inlier_mask],
            inlier_ratio=inlier_ratio,
            essential_matrix=E,
            relative_R=R_rel,
            relative_t=t_rel,
        )

    def build_feature_tracks(
        self,
        features_list: List[ImageFeatures],
        matches_list: List[PairwiseMatch]
    ) -> List[FeatureTrack]:
        """
        Disjoint-set union-find algorithm to chain 2D pairwise matches
        into multi-view continuous feature tracks.
        """
        # Node identifier: (image_idx, feature_idx)
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def find(node: Tuple[int, int]) -> Tuple[int, int]:
            if node not in parent:
                parent[node] = node
                return node
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1: Tuple[int, int], node2: Tuple[int, int]):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                parent[root2] = root1

        # Union corresponding features
        for match in matches_list:
            for s_idx, d_idx in zip(match.inlier_indices_src, match.inlier_indices_dst):
                union((match.src_idx, int(s_idx)), (match.dst_idx, int(d_idx)))

        # Group observations by track root
        track_groups: Dict[Tuple[int, int], Dict[int, Tuple[float, float]]] = {}
        track_colors: Dict[Tuple[int, int], List[np.ndarray]] = {}

        for feat in features_list:
            img_idx = feat.image_idx
            for f_idx, (pt, col) in enumerate(zip(feat.keypoints, feat.colors)):
                node = (img_idx, f_idx)
                if node in parent:
                    root = find(node)
                    if root not in track_groups:
                        track_groups[root] = {}
                        track_colors[root] = []
                    track_groups[root][img_idx] = (float(pt[0]), float(pt[1]))
                    track_colors[root].append(col)

        # Build feature track objects (keeping tracks with length >= 2)
        feature_tracks: List[FeatureTrack] = []
        track_id = 0
        for root, obs in track_groups.items():
            if len(obs) >= 2:
                # Average color across observations
                cols = track_colors[root]
                mean_color = np.mean(cols, axis=0).astype(np.uint8) if cols else np.array([128, 128, 128], dtype=np.uint8)
                feature_tracks.append(FeatureTrack(
                    track_id=track_id,
                    observations=obs,
                    color=mean_color
                ))
                track_id += 1

        return feature_tracks
