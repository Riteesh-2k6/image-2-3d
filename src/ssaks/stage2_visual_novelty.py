"""
SSAKS Stage 2: Visual Novelty & Embedding Clustering
====================================================
Computes normalized visual feature embeddings (DINOv2 / lightweight visual encoder)
and selects frames exhibiting significant visual perspective change (cosine similarity < threshold).
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from src.ssaks.types import SSAKSConfig


class VisualNoveltyClusterer:
    """Stage 2 filter ensuring viewpoint diversity and visual novelty."""

    def __init__(self, config: Optional[SSAKSConfig] = None):
        self.cfg = config or SSAKSConfig()
        self.keyframe_embeddings: List[np.ndarray] = []

    def extract_embedding(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Extract normalized spatial geometry descriptor representation.
        Uses a 32x32 normalized spatial luminance patch to capture perspective and viewpoint changes.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY) if len(bgr_frame.shape) == 3 else bgr_frame
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Zero-mean unit-variance normalization
        thumb = (thumb - np.mean(thumb)) / (np.std(thumb) + 1e-6)
        emb = thumb.flatten()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-7)

    def evaluate_novelty(self, bgr_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate if the frame introduces sufficient visual novelty compared to previous keyframes.
        Returns: (is_novel, max_cosine_similarity)
        """
        emb = self.extract_embedding(bgr_frame)
        
        if not self.keyframe_embeddings:
            self.keyframe_embeddings.append(emb)
            return True, 0.0

        # Compute cosine similarity against recent keyframes (last 5)
        recent_embs = np.stack(self.keyframe_embeddings[-5:], axis=0)
        similarities = np.dot(recent_embs, emb)
        max_sim = float(np.max(similarities))

        if max_sim < self.cfg.dino_similarity_thresh:
            self.keyframe_embeddings.append(emb)
            return True, max_sim

        return False, max_sim
