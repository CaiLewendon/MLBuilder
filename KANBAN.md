# Kanban

## Done
- Established Pi + Coral inference bring-up end-to-end.
- Confirmed correct model artifact for TPU deployment.
- Diagnosed and fixed raw-head decode routing bug (`(5,8400)` vs `[N,6]`).
- Verified relay outage as independent failure source from inference.
- Added first-pass runtime quality mitigations:
  - geometric post-filtering
  - optional crop-pass strategy
- Verified detections can be produced continuously after decode fix.

## In Progress
- Runtime tuning to reduce false positives while preserving far-target recall.

## Next (High Priority)
1. Add deterministic debug counters in production script:
   - `raw_count`
   - `filtered_count`
   - `crop1_added`
   - `crop2_added` (if enabled)
2. Lock one "permissive baseline" config that always detects something in current scene.
3. Incrementally increase strictness to remove false positives:
   - raise `min_conf` gradually
   - apply/adjust area-edge gates
   - test optional aspect-ratio gate after baseline is stable
4. Validate far-target recall with upper-biased crop center:
   - center crop and second upper crop
5. Save one known-good argument profile in `lteboardroutingcommand.txt`.

## Next (Medium Priority)
1. Add temporal confirmation gate (2-of-3 frame persistence) as optional argument.
2. Add CLI flags for all filter thresholds and crop centers/ratios.
3. Add a "debug overlay mode" to color boxes by source pass (full/crop1/crop2).

## Backlog
1. Retraining path for durable accuracy improvement:
   - more far-distance positives
   - hard-negative clutter examples
2. Add regression tests for decoder branch selection in `TFLiteModel.detect()`.
3. Add deployment health checks:
   - relay alive check
   - receiver caps consistency check

## Blockers / Risks
- Single-class model in cluttered scene has intrinsic ambiguity at long distance.
- Over-filtering currently causes all detections to disappear in some configs.
- Lighting and perspective variation likely exceed model robustness without retraining.
