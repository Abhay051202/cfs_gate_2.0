# GateVision runtime contract

Updated: 29 October 2026, 01:12 IST. This describes the current implementation,
including its limits, rather than a claim of universal accuracy.

## Required output

Every supported container identity must survive finalization. Twin classification
is optional: two identities without proven chassis association become separate
records. Competing readings of one text region require review. Each truck visit
owns one shared image folder; cameras are observers, not event owners.

Overview proof and OCR evidence are separate. Overview targets a repeatable
physical capture position and maximum useful container visibility. The printed
number need not be readable in the overview. OCR may use another frame.

## Flow

1. Camera workers continuously publish frames. A bounded history defaults to
   12 frames and 24 MB per camera, in addition to the latest-frame inference slot.
2. YOLO detects containers/text and the auxiliary vehicle detector supplies
   truck/chassis observations. Tracking assigns lane/view ownership.
3. The collector can keep multiple visits open. A following truck cannot borrow
   a visit already observed in that camera. Presence of a different truck does
   not renew an older truck's occupancy.
4. The overview coordinator observes measured visits independently of OCR.
   At the reference capture line it selects buffered frames near one host receive
   time. The selected pixels are checked for container evidence. Stale frames,
   old source sessions and unassigned views during competing visits are held.
5. OCR independently samples identification regions. The final planner preserves
   each complete checksum-valid observed identity and isolates its evidence.
   A checksum alone does not establish correctness. Strict confirmation requires
   matching independent frames from two cameras at confidence >= 0.90.
6. After departure/clearance and the configured reversal grace, plans are saved.
   A failure in one visit is isolated, retried at most three times with backoff,
   then exposed for review rather than repeatedly blocking the entire OCR loop.
7. The preview builder reads the overview manifest and verifies image hashes.
   Complete, confirmed payloads are frozen. UI and upload builder use that same
   frozen payload. Incomplete proof is visible locally but cannot be uploaded.

Separate uncertain-container records retain the shared local visit folder but
use distinct event-based cloud VisitId keys. This prevents the cloud treating
the second record as a retry of the first. A proven twin is one record with two
containers. The internal source visit ID is retained in the preview audit data.

## Fixed proof calibration

Use Lane calibration → Overview reference on one camera per gate. Choose a
vertical/horizontal capture line, normalized position, and movement direction.
Saving a new reference clears other references on that gate. The reference is
the detected target's center, not an axle detector or a hardware light beam.
Old "laser trigger" labels in the legacy editor are not physical sensors.

Without this calibration the coordinator retains a maximum-occupancy fallback
and displays NEEDS_CALIBRATION. It does not claim repeatable position or permit
upload. If a camera misses the trigger or visibility/ownership is uncertain,
the captured set is retained for review; the trigger is not silently moved.

Alignment uses HOST_RECEIVE_TIME with a default 0.3-second bound. It is not a
guarantee of simultaneous camera exposures. Fast crossings use an interpolated
trigger time between adjacent tracked positions (only across gaps <= 1 second),
then select real buffered images. No synthetic image is generated. Site testing
must establish whether stream delay, FPS and placement meet the client's spatial
tolerance. The code cannot infer a physical capture position after a camera moves.

## Approved PUT proof updates

The event detail page lists clearer full-frame candidates near the fixed trigger.
An operator can review and select a replacement. The backend verifies event
ownership, capture time, dimensions, quality within the target region, current
container visibility and measured position drift/IoU before publishing a revision.
Later OCR frames cannot become overview replacements merely because text is sharper.

`POST /api/gate-events/{id}/proof-revisions` accepts snapshot_id and a review reason.
It archives the previous frozen payload, uses a compare-and-swap digest to reject
stale reviews, persists the new pending revision and refreshes the local proof.
For a proven twin, identical shared camera overview images are revised together.
Other camera images remain unchanged. The original trigger stays fixed.

In coordinated mode, `upgrade-proofs` sends only the complete approved payload
using `PUT /api/v1/gate/events/{cloud-event-id}/images`. It does not scan arbitrary
files or integrate live OCR candidates. Automatic dispatch settings can queue an
approved revision; local-test mode blocks the worker and all PUT requests while
allowing local review. The manual dispatch path also uses the frozen payload.

Unknown size remains null instead of defaulting to 20 ft. Confidence values and
encoded image bytes are preserved. Duplicate/missing camera images are rejected.
Offline, rejected, conflicting or unsuccessful responses do not acknowledge the
new revision. Retries reuse its bytes/version; an older in-flight acknowledgment
cannot mark a newer revision synced or overwrite newer acknowledged proof rows.
Camera/container coverage cannot shrink through this updater.

The saved Swagger specification does not describe this PUT endpoint. Its existing
application implementation was checked with mocked requests; no real server PUT
was issued during this work. Server-side behavior still requires staging validation.

## Storage and settings

Live data normally uses backend/data/gate.db, backend/snapshots and
backend/uploads/ocr-spool. All visit camera images share the visit folder.
OCR job metadata stays in SQLite; folder co-location does not itself prove
truck ownership. Temporary OCR samples are bounded/pruned, not a promise to
retain every frame forever.

Within the visit folder, overview.json records the reference, trigger time,
required/missing cameras, source sessions, target boxes, hashes and state.
camera_overview.jpg files are distinct from OCR proof frames. Finalized folders
are promoted out of waiting. The exact encoded outgoing payload is retained in
snapshots/outgoing-preview/<event-id>.json with an integrity digest.

New settings: PROOF_CAPTURE_ENABLED, PROOF_SYNC_TOLERANCE_SECONDS,
PROOF_BUFFER_FRAMES, PROOF_BUFFER_MB. Per-camera calibration adds
proof_capture_axis, proof_capture_position, proof_capture_positive.

## Safe local validation

From the repository root on Windows:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\local_test.py --process --device cuda:0
```

Open http://127.0.0.1:8002/. The launcher unconditionally disables cloud access
and dispatch, uses backend/local-test for data/images/logs, and reuses only camera
connection settings plus initial operator camera preferences. It does not clear
existing live data. CPU is available with --device cpu; CPU OCR remains explicitly
configured in this Windows test launcher. No TensorRT/GPU OCR claim is implied.

## Acceptance and remaining limits

Regression tests cover identity retention, uncertain twins, colliding checksums,
successive trucks, stale/missing views, immutable proof, upload exclusion and API
reporting. Synthetic tests establish logic, not OCR or tracking accuracy on site.

Run tools/evaluate_ocr.py against independently labelled image crops to measure
exact OCR accuracy and day/night/blur buckets. CSV columns are
image,container_number,condition. Include unreadable and wrong results in the
denominator. This measures image OCR, not complete visit association or proof
quality. The 90%+ target has not yet been demonstrated by these changes.

Before enabling cloud: calibrate the reference line, replay labelled local
footage with bumper-to-bumper traffic and different speeds, inspect proof
visibility and timing, and verify the Prosper contract for separately retained
container records sharing one physical visit. Occlusion, similar trucks and
uncalibrated cross-camera travel can remain ambiguous; such cases must stay in
review rather than be silently treated as proven associations.

Persisted evidence is recoverable after restart; full physical tracking state is
not persisted as a durable gate state machine. Hardware synchronization,
automatic site calibration and measured 90%+ event-level accuracy are not
implemented guarantees.
