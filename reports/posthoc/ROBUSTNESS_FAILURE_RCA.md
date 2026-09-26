# Robustness Failure Root Cause

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

All results reuse the 14 existing immutable robustness outputs; no robustness inference was rerun. The candidate explanation that the learned router may conflate missing-content or cold-item signals with a reason to increase content reliance is classified **WEAK_ASSOCIATION**. Mean content-reliance change for zero-or-low-support rows across missingness cells was 0.001324.

This classification describes observed routing deltas. It does not establish a causal mechanism. Covariate-specific associations are labeled SUPPORTED_ASSOCIATION, WEAK_ASSOCIATION, NOT_SUPPORTED, or UNTESTABLE in the machine-readable summary.
