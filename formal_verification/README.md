# Lean 4 Formal Safety Verification (`GlidanceSafety`)

This directory contains the machine-checked **Lean 4 (v4.35.0)** formal specification and proofs for Glidance's safety-critical navigation & V-disparity geometry kernel, mirrored 1:1 in [`rust_core/src/lib.rs`](../rust_core/src/lib.rs) (compiled to both native `cdylib` and mobile `wasm32-unknown-unknown`).

## Kernel Verification Status
- **Declarations & Theorems Checked:** `149` (`34` theorems: `6` non-vacuous satisfiability witnesses + `28` safety, projective-geometry, and hysteresis invariants)
- **`sorryAx` Count:** `0`
- **Custom Axioms:** `0`
- **`unsafe` / `partial` Declarations:** `0`
- **Checker:** `SafeVerify` + `leanchecker` (`EXIT 0`)

## Key Proven Safety Invariants
1. **Fail-Closed Telemetry (`decideGuidance_failClosed`):** Stale frames or dropped camera/WebSocket connections (`frameFresh = false`) deterministically command `STOP` with `overallSeverity = danger` (never leaving a blind walker on `PATH CLEAR`).
2. **Strict Severity Dominance (`decideGuidance_severity_dominance`, `decideGuidance_severity_exact`):** `overallSeverity` is proven to equal the exact lattice join $\max(L, C, R, \text{curb})$ on fresh frames, eliminating severity-downgrade bugs when center is `caution` while a side corridor is `danger`.
3. **Never Steer Into Danger (`decideGuidance_never_veer_left_danger`, `decideGuidance_never_veer_right_danger`, `decideGuidance_all_danger_stop`):** `VEER_LEFT` mathematically implies `left.severity ≠ danger`; `VEER_RIGHT` implies `right.severity ≠ danger`; full corridor blockage forces `STOP`.
4. **Optimal Safe Escape & Symmetric Stability (`decideGuidance_escape_exists`, `decideGuidance_escape_prefers_lower_severity`, `decideGuidance_symmetric_caution_centered`):** Never freezes when a safe lateral corridor exists, strictly prefers `clear` over `caution`, and eliminates asymmetric right-bias on uniform caution walls.
5. **Tilted Floor Zero-False-Alarm & Wall-Collapse Override (`flatFloor_profile_zero_excess`, `wallOverride_uniform_close_wall`, `wallOverride_full_corridor_blockage`):** Bounded tilted floors produce identically zero obstacle excess, while uniform close walls (`spread ≤ 80‰` with `upperMedian ≥ 700‰`, or `upperMedian ≥ 840‰`) trigger `wallDangerScore = 1000‰` even when relative contrast collapses.
6. **Curb-vs-Obstacle Mutual Exclusion (`curb_obstacle_mutual_exclusion`, `isTrueCurbDrop_sound`):** Any upright low obstacle/crate protruding above the floor prior (`fgExcess > 120‰`) is proven incapable of falsely triggering `CURB_DROP`.
7. **EMA Convex Hull & Asymmetric Streak Hysteresis (`emaStep_convex`, `shortBurst_not_confirmed`, `consecutiveHazard_confirms`, `classifyZone_hysteresis`):** Fixed-point EMA never overshoots `[min, max]` or `[0..1000]`, transient noise spikes `< confirmFrames` warn as `caution` without false `danger`, and `confirmFrames` consecutive hazard frames deterministically confirm `danger`.
