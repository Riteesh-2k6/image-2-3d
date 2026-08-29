# SSAKS Emergency Fallback

If no stabilized flight segment is detected within 30 seconds, SSAKS will fall back to motion-adaptive sampling (optical flow + blur only), followed by DINO novelty if possible.

We explicitly reject falling back to a fixed FPS sampling rate as the primary fallback, because fixed FPS increases redundant frames and threatens the strict 6 GB VRAM budget. Fixed FPS is relegated to the final emergency fallback, and any fallback from a stabilized cruise will emit a reconstruction quality warning to the operator.
