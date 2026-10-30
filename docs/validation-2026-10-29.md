# Validation — 29 October 2026

- Main backend regression run: **305 passed, 5 skipped**, 53.80 seconds.
  The real-image Paddle integration test was excluded; external MySQL was
  disabled. These results do not measure real-world OCR accuracy.
- After the final cloud-key/nullable-size change: **51 proof, payload, OCR
  acceptance and local-only tests passed**, 12.77 seconds.
- Angular production build passed. Python compilation and git diff checks passed.
- The isolated runtime loaded detector and OCR successfully. Cloud uploads were
  disabled, local-test mode enabled, and no processing errors or held visits
  were reported. All cameras were offline, so health correctly showed DEGRADED.
- No cloud events were sent. No commit/push was performed.

Remaining acceptance work: one reference-line calibration per gate, connected
camera/replay checks of actual truck handoffs and proof position, and exact-match
OCR evaluation against independently labelled day/night/blurred images. The
90%+ target is not established by unit tests or by OCR confidence scores.

## Approved proof-update validation

Final backend regression: **324 passed, 5 skipped**, 103.19 seconds. Real-image
Paddle integration remained excluded and external MySQL disabled. Existing
Starlette deprecation and legacy snapshot identity-map warnings remain.
Angular production build passed with the proof review controls.

Added tests cover exact UI/PUT payload equality, local-only approval without
upload, unchanged companion-camera images, late/moved-frame rejection, stale
review rejection, revision archival, offline/rejected/conflicting responses,
retry byte/version stability, camera coverage preservation, duplicate-camera
rejection, null unknown size, unrounded confidence, and a newer revision arriving
while an earlier PUT is in flight. HTTP calls were mocked or local only.
