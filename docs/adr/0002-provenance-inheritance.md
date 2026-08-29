# Provenance Inheritance

Provenance will be tracked strictly at the Gaussian primitive level (Observed, Prior-Guided, AI-Inferred). During the SuGaR/SERE extraction phase, each mesh face will inherit its provenance from its contributing Gaussians via weighted aggregation.

This allows a single architectural entity (like a large building) to accurately reflect mixed provenance, containing both Observed and Prior-Guided regions, rather than forcing a coarse classification at the object level.
