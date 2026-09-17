import time
import cv2
import sqlite3
from pathlib import Path
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.snapshots.snapshot_manager import SnapshotManager
from app.runtime import GateRuntime
from app.domain import Frame, utcnow

# 1. Reset tables
for table in ['gate_events', 'detections', 'ocr_results', 'snapshots']:
    with sqlite3.connect('data/gate.db') as conn:
        conn.execute(f'DELETE FROM {table}')
with sqlite3.connect('uploads/ocr-spool/jobs.sqlite3') as conn:
    conn.execute('DELETE FROM jobs')

settings = Settings()
db = Database(settings.database_url.get_secret_value())
repo = GateRepository(db)
snaps = SnapshotManager(settings.snapshot_directory)
runtime = GateRuntime(settings, settings.cameras(), repo, snaps)

# 2. Start runtime
runtime._initialize_pipeline()
runtime.start()

video_path = 'uploads/b1a6d126d7a340e7bf1d7af07745a1a5.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Starting test on {video_path}: {total} frames at {fps:.1f} fps...')

camera = settings.cameras()[0]
frame_idx = 0
step = 4  # sample every 4th frame (~6 fps)

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if frame_idx % step == 0:
        f = Frame(camera.id, utcnow(), frame_idx, frame_bgr, source_id='file')
        runtime.process_frame(camera, f)
        if frame_idx % 40 == 0:
            print(f'Processed frame {frame_idx}/{total}...')
    frame_idx += 1

cap.release()
print('Video playback completed. Waiting for OCR queue to process...')

# Wait up to 30 seconds for OCR spool to finish
t0 = time.time()
while time.time() - t0 < 30:
    time.sleep(1)
    evs = repo.events(limit=5).get('items', [])
    if evs:
        ev = evs[0]
        if ev.get('is_twin') and ev.get('container_number_2'):
            print(f'SUCCESS! Twin container confirmed in {time.time() - t0:.1f}s!')
            break

runtime.stop()

# Query final event
print('\n=== FINAL EVENTS IN DATABASE ===')
with sqlite3.connect('data/gate.db') as conn:
    conn.row_factory = sqlite3.Row
    for row in conn.execute('SELECT id, is_twin, container_number, container_number_2, iso_code_1, iso_code_2, status FROM gate_events').fetchall():
        print(dict(row))

with sqlite3.connect('uploads/ocr-spool/jobs.sqlite3') as conn:
    conn.row_factory = sqlite3.Row
    print('\n=== SPOOL JOBS ===')
    for row in conn.execute('SELECT id, origin, sequence, state, validation, event_id FROM jobs').fetchall():
        print(dict(row))
