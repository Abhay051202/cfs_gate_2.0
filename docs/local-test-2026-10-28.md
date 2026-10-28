# Local test session — 28 October 2026

Repository: cfs_gate_2.0 Edge_box, base b3b8069.
Local checkout: D:/ai-video-based-inventory/cfs-gate-2-local-test.
URL: http://127.0.0.1:8002/.

Run from backend using the existing dependency interpreter:

```powershell
& D:/ai-video-based-inventory/ai-video-based-inventory/backend/.venv/Scripts/python.exe local_test.py --process
```

The launcher forces local_test_mode, disables cloud and auto-dispatch, imports
only camera URLs/ONVIF credentials from the original .env, and keeps runtime
records in this checkout. It reuses original model files without changing them.
RapidOCR is installed privately under backend/.local-vendor. YOLO is configured
for CUDA; OCR explicitly uses CPU because this test has no Windows TensorRT OCR
engines. The UI was built from this checkout.

The cloud dispatcher is disabled, resume cannot enable it, direct session POST/
PUT/login paths are blocked, and a transport guard also prevents external HTTP
through the Prosper session. Proof-upgrade processing is disabled in local mode.
No cloud upload was made. Five streams were observed online; gate-out-3 is disabled.

The first confirmed upload preview is saved with encoded image bytes and a
payload checksum under backend/snapshots/outgoing-preview/{event_id}.json.
The API preview and future upload builder load this same payload. Incomplete
events are not frozen. The UI labels local-only testing and disables Send to Site.
This freezes the upload preview, not the physical vehicle position. Coordinated
camera capture and axle triggering are not implemented or validated here.

Validation: initial cloud/preview test run 26 passed; after adding the explicit
review revision, all seven local-only tests passed. Angular production build
passed. Browser inspection confirmed the local-only banner, disabled upload
button and three proof images. No claim is made that the entire release suite
or multi-camera synchronization is validated.

## Reviewed Gate IN 3 revision

Event: 8e6d701c-c73a-593c-9927-470d5e294aee, container TGBU3796155.
Visit: 835d3a56-1aca-44a7-9e87-d4385523876d.

The initial gate-in-3 proof came from the first side-number OCR frame at backend
time 08:33:35.858136 UTC. A retained original context frame at 08:33:41.257566 UTC
shows the rear doors and horizontal rear marking. Its source metadata matches
the same visit and camera, and the container marking was visually checked.
The user requested an improved gate-3 image. Only that camera's frozen preview
was explicitly revised to the original context_1.jpg, with no visual editing,
cropping, sharpening, synthetic content or new OCR confidence.

The previous envelope is archived as {event_id}.before-v2.json. Revision 2 includes
a review reason and source path. Cameras 1/2 image hashes were verified unchanged;
gate-3's new hash matches the original JPEG exactly. Browser verification shows
the revised gate-3 image and capture time 14:03:41 IST. The camera OSD and backend
timestamps differ slightly; no synchronization claim is inferred from them.

This is a single reviewed correction, not an automatic rear-door selection fix
for future events. Automatic selection still needs calibration/trigger work.
