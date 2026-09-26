# Executive Summary

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

Canonical p2 retained its baseline. This read-only study finds that R4's controlled-zero advantage was sparse and concentrated, while warm and high-support slices continued to favor C_seq. R4 achieved 164.97% of max-per-query expert NDCG because its distinct ranking sometimes exceeded both experts; this ratio is not a true oracle bound. The absolute expert quality and localized routing/fusion errors indicate weak experts overall, with localized routing/fusion errors. Catalog health and temporal behavior are reported as offline diagnostics only.

No posthoc finding revises p2. The useful outcome is a p3 hypothesis set: hard zero-support behavior, explicit warm preservation, stronger score calibration, and validation-only routing supervision.
