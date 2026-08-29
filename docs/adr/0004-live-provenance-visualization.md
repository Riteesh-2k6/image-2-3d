# Live Provenance Visualization

Live Mode will strictly render provenance visually to match the final export confidence. Observed regions will render solid, Prior-Guided as translucent/wireframe, and AI-Inferred as dotted/hatched.

This guarantees a WYSIWYG (What You See Is What You Get) reconstruction confidence during flight, preventing a jarring UX where buildings appear in Live Mode but disappear in the final export due to confidence filtering.
