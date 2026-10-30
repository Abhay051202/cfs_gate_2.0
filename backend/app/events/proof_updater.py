"""
backend/app/events/proof_updater.py

Automated Cloud Proof Upgrader / Proof Updater Module.
Detects when superior, higher-quality, or late container proof images arrive
for an already-synced Gate Event, promotes the superior images locally, and
automatically updates the Prosper Smart Yard cloud via PUT /api/v1/gate/events/{cloud_event_id}/images.

Strictly adheres to Prosper Cloud API constraints:
1. Unique CameraId per container (avoids HTTP 400 ValidationException).
2. Slot mapping separation (FRONT vs REAR) so container proof cameras (e.g. gate-in-2, gate-out-1)
   are never overwritten by cabin/approach context cameras in Azure Blob storage.
3. Cumulative acknowledged proof retention (never shrinks verified proof sets).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from app.database.models import utcnow
from app.events.cloud_dispatcher import (
    _iso_utc,
    resolve_camera_image_type,
    resolve_container_size,
)
from app.snapshots.proof import proof_jpeg
from app.snapshots.outgoing_preview import OutgoingPreview

log = logging.getLogger('gate.proof_updater')


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _parse_ts(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if not val:
        return utcnow()
    try:
        text = str(val).replace('Z', '+00:00')
        return datetime.fromisoformat(text)
    except Exception:
        return utcnow()


class CloudProofUpgrader:
    """
    Dedicated manager for evaluating and upgrading cloud proof images.
    Can run asynchronously via an internal background queue or synchronously.
    """

    def __init__(
        self,
        settings: Any,
        repository: Any = None,
        snapshots: Any = None,
        dispatcher: Any = None,
        session: Any = None,
        runtime: Any = None,
    ):
        self.settings = settings
        self.repository = repository
        self.snapshots = snapshots
        self.dispatcher = dispatcher
        self.session = session or getattr(dispatcher, 'session', None)
        self.runtime = runtime

        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the background upgrade worker thread."""
        if getattr(self.settings, 'local_test_mode', False) is True:
            return
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name='gate-cloud-proof-upgrader',
                daemon=True,
            )
            self._worker_thread.start()
            log.info('cloud_proof_upgrader_started')

    def stop(self, timeout: float = 2.0):
        """Stop the background upgrade worker thread."""
        self._stop_event.set()
        thread = self._worker_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._worker_thread = None
        log.info('cloud_proof_upgrader_stopped')

    def enqueue_upgrade(
        self,
        event_id: str,
        candidate_image: Optional[Any] = None,
        camera_id: Optional[str] = None,
        ocr_result: Optional[Any] = None,
        candidate_path: Optional[str | Path] = None,
        force: bool = False,
    ):
        """
        Enqueue an event for proof evaluation and cloud upgrade.
        Non-blocking, safe to call from live camera inference / OCR worker threads.
        """
        if not event_id:
            return
        if getattr(self.settings, 'local_test_mode', False) is True:
            return
        item = {
            'event_id': event_id,
            'candidate_image': candidate_image,
            'camera_id': camera_id,
            'ocr_result': ocr_result,
            'candidate_path': candidate_path,
            'force': force,
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            log.warning('cloud_proof_upgrader_queue_full: dropping upgrade check for %s', event_id)

    def _worker_loop(self):
        """Internal background loop consuming upgrade tasks."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self.evaluate_and_upgrade(
                    event_id=item['event_id'],
                    candidate_image=item.get('candidate_image'),
                    camera_id=item.get('camera_id'),
                    ocr_result=item.get('ocr_result'),
                    candidate_path=item.get('candidate_path'),
                    force=item.get('force', False),
                )
            except Exception as exc:
                log.exception('cloud_proof_upgrader_task_error: %s', exc)
            finally:
                self._queue.task_done()

    def evaluate_and_upgrade(
        self,
        event_id: str,
        candidate_image: Optional[Any] = None,
        camera_id: Optional[str] = None,
        ocr_result: Optional[Any] = None,
        candidate_path: Optional[str | Path] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate proof images for an event. If a higher quality or previously missing
        container proof is found, update the event in SQLite and dispatch via PUT /images.
        """
        if not self.repository:
            return {'success': False, 'status': 'NO_REPOSITORY'}
        if getattr(self.settings, 'local_test_mode', False) is True:
            return {'success': False, 'status': 'LOCAL_ONLY'}

        event_record = self.repository.event(event_id)
        if not event_record:
            return {'success': False, 'status': 'NOT_FOUND', 'error': f'Event {event_id} not found'}

        cloud_event_id = event_record.get('cloud_event_id')
        cloud_synced = bool(event_record.get('cloud_synced'))

        # Only already-synced events with a valid cloud_event_id receive PUT /images updates.
        # (Unsynced events will be dispatched via POST /events by normal dispatcher).
        coordinated = getattr(self.settings, 'proof_capture_enabled', False) is True
        if not cloud_event_id or (not cloud_synced and not coordinated):
            return {
                'success': False,
                'status': 'NOT_ELIGIBLE',
                'reason': 'Event is not yet synced to cloud or lacks cloud_event_id',
            }

        session = self.session or (getattr(self.dispatcher, 'session', None) if self.dispatcher else None)
        if not session or not hasattr(session, 'put_gate_event_images'):
            return {'success': False, 'status': 'NO_SESSION', 'error': 'Cloud session with put_gate_event_images unavailable'}

        if coordinated:
            # Live OCR candidates cannot choose a new camera position. Only a
            # persisted, approved revision is eligible for this PUT path.
            return self._sync_approved_revision(event_record, session)

        # 1. Promote / store candidate image if provided
        if candidate_image is not None and camera_id and self.snapshots:
            self._integrate_candidate_image(event_record, candidate_image, camera_id, ocr_result)

        # 2. Check and promote waiting directory / unattached frames on disk
        self._scan_and_promote_unattached_frames(event_record)

        # 3. Reload event record to reflect newly saved snapshots
        event_record = self.repository.event(event_id) or event_record

        # 4. Check if current candidate proof set improves upon acknowledged proofs
        improved, candidate_snapshots = self._evaluate_proof_improvement(event_record, force=force)
        if not improved:
            return {
                'success': True,
                'status': 'NOOP',
                'message': 'Acknowledged proofs are already up to date with best available captures',
            }

        # 5. Build PUT payload respecting unique CameraId & collision-free ImageType
        try:
            payload = self._build_put_payload(event_record, candidate_snapshots)
        except Exception as exc:
            log.warning('cloud_proof_upgrader_payload_failed: %s', exc)
            return {'success': False, 'status': 'PAYLOAD_ERROR', 'error': str(exc)}

        # 6. Execute PUT /api/v1/gate/events/{cloud_event_id}/images
        res = session.put_gate_event_images(cloud_event_id, payload)
        if res is True or (isinstance(res, dict) and res.get('success') is not False):
            new_version = int(event_record.get('proof_version', 1) or 1) + 1
            try:
                self.repository.mark_cloud_synced(event_id, cloud_event_id=cloud_event_id, proof_version=new_version)
                self._record_payload_proofs(event_id, cloud_event_id, payload, new_version)
            except Exception as exc:
                log.warning('cloud_proof_upgrader_ack_persistence_failed: %s', exc)

            log.info('prosper_cloud_proof_upgrade_success', extra={
                'event_id': event_id,
                'cloud_event_id': cloud_event_id,
                'proof_version': new_version,
            })
            return {
                'success': True,
                'status': 'UPGRADED',
                'cloud_event_id': cloud_event_id,
                'proof_version': new_version,
                'containers_count': len(payload.get('Containers', [])),
            }
        elif res == 'OFFLINE':
            log.info('prosper_cloud_proof_upgrade_offline: event %s buffered', event_id)
            return {'success': False, 'status': 'OFFLINE', 'error': 'Cloud service currently offline'}
        else:
            log.warning('prosper_cloud_proof_upgrade_rejected: %s', res, extra={'event_id': event_id, 'status': str(res)})
            return {'success': False, 'status': str(res), 'error': f'Cloud returned status: {res}'}

    def _sync_approved_revision(self, event_record, session):
        event_id = event_record['id']
        if event_record.get('status') != 'CONFIRMED':
            return {'success': False, 'status': 'NEEDS_REVIEW'}
        store = OutgoingPreview(self.snapshots.root)
        try:
            payload = store.load(event_id)
            if not payload or not payload.get('_proof_complete') or not payload.get('_overview', {}).get('frozen'):
                return {'success': False, 'status': 'NO_APPROVED_PROOF'}
            if self.dispatcher:
                approved = self.dispatcher.build_event_payload(event_id)
                if store.digest(approved) != store.digest(payload):
                    raise ValueError('Proof revision changed during payload selection')
        except (ValueError, OSError) as exc:
            return {'success': False, 'status': 'PAYLOAD_ERROR', 'error': str(exc)}
        version = int(payload.get('ProofVersion') or 1)
        acknowledged = self.repository.get_acknowledged_proofs(event_id)
        expected = {(c['ContainerNumber'], image['CameraId']): image['_image_hash']
                    for c in payload['Containers'] for image in c['Images']}
        actual = {(p['container_number'], p['camera_id']): p['image_hash'] for p in acknowledged}
        if expected == actual and all(p.get('proof_version') == version for p in acknowledged):
            self.repository.mark_cloud_synced(event_id, cloud_event_id=event_record['cloud_event_id'], proof_version=version)
            return {'success': True, 'status': 'NOOP', 'proof_version': version}
        # Never shrink or change the container/camera identities already shown
        # by the cloud. A different event identity requires explicit handling.
        if not actual.keys() <= expected.keys():
            return {'success': False, 'status': 'PROOF_COVERAGE_REGRESSION'}
        self.repository.mark_proof_pending(event_id, version)
        result = session.put_gate_event_images(event_record['cloud_event_id'], payload)
        if result is not True and not (isinstance(result, dict) and result.get('success') is not False):
            return {'success': False, 'status': str(result)}
        # Record precisely what the server accepted, even if another local
        # revision was approved while this request was in flight.
        self._record_payload_proofs(event_id, event_record['cloud_event_id'], payload, version)
        current = store.load(event_id)
        if not current or store.digest(current) != store.digest(payload):
            if current:
                self.repository.mark_proof_pending(event_id, int(current.get('ProofVersion') or version))
            return {'success': True, 'status': 'NEWER_REVISION_PENDING', 'proof_version': version}
        self.repository.mark_cloud_synced(event_id, cloud_event_id=event_record['cloud_event_id'], proof_version=version)
        return {'success': True, 'status': 'UPGRADED', 'proof_version': version,
                'cloud_event_id': event_record['cloud_event_id'], 'payload_sha256': store.digest(payload)}

    def _integrate_candidate_image(
        self,
        event_record: Dict[str, Any],
        candidate_image: Any,
        camera_id: str,
        ocr_result: Optional[Any] = None,
    ):
        """Save candidate image as snapshot if it is valid and adds value."""
        try:
            event_id = event_record['id']
            direction = event_record.get('event_type') or 'ENTRY'
            snap_id = str(uuid4())

            img = candidate_image
            if hasattr(img, 'image'):
                img = img.image
            if not isinstance(img, np.ndarray) or img.size == 0:
                return

            rel_path = self.snapshots.save_side_snapshot(
                event_id,
                f'{camera_id}_{snap_id}_proof',
                img,
                direction=direction,
                force=True,
            )
            if hasattr(self.repository, 'add_or_update_snapshot'):
                self.repository.add_or_update_snapshot(event_id, camera_id, rel_path, utcnow())
        except Exception as exc:
            log.debug('integrate_candidate_image_failed: %s', exc)

    def _scan_and_promote_unattached_frames(self, event_record: Dict[str, Any]):
        """Scan waiting directory and event directory for clear proof frames not yet in SQLite snapshots."""
        if not self.snapshots:
            return

        event_id = event_record['id']
        visit_id = event_record.get('visit_id') or event_id
        direction = event_record.get('event_type') or 'ENTRY'

        try:
            if hasattr(self.snapshots, 'promote_from_waiting'):
                self.snapshots.promote_from_waiting(visit_id, direction)
        except Exception:
            pass

        cat = self.snapshots.category_for(direction) if hasattr(self.snapshots, 'category_for') else 'detected_in'
        event_dir = self.snapshots.root / cat / visit_id
        if not event_dir.is_dir():
            event_dir = self.snapshots.root / visit_id
        if not event_dir.is_dir():
            return

        existing_snaps = event_record.get('snapshots') or []
        existing_paths = {s.get('image_path') for s in existing_snaps if s.get('image_path')}

        # Search for proof/frame candidates (e.g. gate-in-2_*_frame.png or gate-in-2_*_proof.jpg)
        for frame_file in sorted(event_dir.glob('*_frame.png')) + sorted(event_dir.glob('*_proof.jpg')):
            cam_prefix = frame_file.stem.split('_')[0]
            rel_path = f'{cat}/{visit_id}/{frame_file.name}'
            if rel_path in existing_paths or str(frame_file) in existing_paths:
                continue

            try:
                data = frame_file.read_bytes()
                decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[0] < 240 or decoded.shape[1] < 320:
                    continue
                q = self.snapshots.compute_quality(decoded)
                if q < 1.8:
                    continue

                # Promote this high-quality frame to SQLite snapshots
                snap_id = str(uuid4())
                durable_name = f'{cam_prefix}_{snap_id}_proof.jpg'
                durable_path = event_dir / durable_name
                shutil.copyfile(str(frame_file), str(durable_path))
                durable_rel = f'{cat}/{visit_id}/{durable_name}'

                if hasattr(self.repository, 'add_or_update_snapshot'):
                    self.repository.add_or_update_snapshot(event_id, cam_prefix, durable_rel, utcnow())
                    existing_paths.add(durable_rel)
            except Exception as exc:
                log.debug('promote_unattached_frame_failed: %s', exc)

    def _evaluate_proof_improvement(
        self,
        event_record: Dict[str, Any],
        force: bool = False,
    ) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Check if the best available snapshots improve upon the currently acknowledged proofs.
        Returns (is_improved, best_snapshots_by_container).
        """
        event_id = event_record['id']
        cntr1 = event_record.get('container_number') or event_record.get('historical_container_number')
        cntr2 = event_record.get('container_number_2') or event_record.get('historical_container_number_2')
        is_twin = bool(event_record.get('is_twin')) and bool(cntr2)

        acked_all = self.repository.get_acknowledged_proofs(event_id) if hasattr(self.repository, 'get_acknowledged_proofs') else []
        acked_hashes_c1 = {a['camera_id']: a.get('image_hash', '') for a in acked_all if not cntr1 or a.get('container_number') == cntr1}
        acked_hashes_c2 = {a['camera_id']: a.get('image_hash', '') for a in acked_all if is_twin and a.get('container_number') == cntr2}

        # Select best snapshot per camera for C1 and C2
        c1_best, c2_best = self._select_best_snapshots(event_record)

        if force:
            return True, {'c1': c1_best, 'c2': c2_best}

        has_improvement = False

        # Compare C1
        for cam_id, snap_info in c1_best.items():
            curr_hash = snap_info['hash']
            acked_hash = acked_hashes_c1.get(cam_id)
            if not acked_hash or curr_hash != acked_hash:
                has_improvement = True
                break

        # Compare C2 if twin
        if not has_improvement and is_twin:
            for cam_id, snap_info in c2_best.items():
                curr_hash = snap_info['hash']
                acked_hash = acked_hashes_c2.get(cam_id)
                if not acked_hash or curr_hash != acked_hash:
                    has_improvement = True
                    break

        return has_improvement, {'c1': c1_best, 'c2': c2_best}

    def _select_best_snapshots(self, event_record: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Group and rank snapshots by camera, separating Container 1 and Container 2 (rear slot)."""
        c1_best: Dict[str, Dict[str, Any]] = {}
        c2_best: Dict[str, Dict[str, Any]] = {}

        snapshots = event_record.get('snapshots') or []
        is_twin = bool(event_record.get('is_twin'))

        for snap in snapshots:
            cam_id = snap.get('camera_id')
            if not cam_id:
                continue

            rel_path = snap.get('image_path')
            if not rel_path:
                continue

            path = self.snapshots.path(rel_path) if self.snapshots else Path(rel_path)
            if not path.is_file():
                continue

            try:
                data = path.read_bytes()
                decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[0] < 240 or decoded.shape[1] < 320:
                    continue

                q = self.snapshots.compute_quality(decoded) if self.snapshots else 2.0
                jpeg_bytes = proof_jpeg(data, snap, event_record)
                img_hash = hashlib.sha256(jpeg_bytes).hexdigest()

                score = float(q)
                if '_proof' in rel_path or 'proof' in str(path):
                    score += 1500.0

                is_rear = is_twin and (
                    rel_path.endswith('_rear.jpg') or
                    snap.get('slot_position') == 'REAR' or
                    'rear' in cam_id.lower() or
                    '-3' in cam_id
                )

                target_dict = c2_best if is_rear else c1_best
                if cam_id not in target_dict or score > target_dict[cam_id]['score']:
                    target_dict[cam_id] = {
                        'snap': snap,
                        'data': jpeg_bytes,
                        'hash': img_hash,
                        'score': score,
                        'quality': q,
                    }
            except Exception as exc:
                log.debug('snapshot_selection_error: %s', exc)

        return c1_best, c2_best

    def _build_put_payload(
        self,
        event_record: Dict[str, Any],
        candidate_snapshots: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Construct the payload conforming to Prosper Smart Yard's
        PUT /api/v1/gate/events/{cloud_event_id}/images endpoint.
        Ensures:
        - Unique CameraId per container.
        - Proper assignment of FRONT vs REAR image types to prevent Azure Blob overwrite.
        """
        event_id = event_record['id']
        visit_id = event_record.get('visit_id') or event_id
        cntr1 = event_record.get('container_number') or event_record.get('historical_container_number') or ''
        cntr2 = event_record.get('container_number_2') or event_record.get('historical_container_number_2') or ''
        is_twin = bool(event_record.get('is_twin')) and bool(cntr2)

        if getattr(self.settings, 'proof_capture_enabled', False) is True:
            payload = OutgoingPreview(self.snapshots.root).load(event_record['id'])
            if not payload or not payload.get('_proof_complete'):
                raise ValueError('No complete approved proof revision')
            return payload
        c1_dict = candidate_snapshots.get('c1', {})
        c2_dict = candidate_snapshots.get('c2', {})

        cam_cfgs = {}
        if hasattr(self.settings, 'cameras'):
            for c in getattr(self.settings, 'cameras', []):
                cid = getattr(c, 'id', None) or (c.get('id') if isinstance(c, dict) else None)
                if cid:
                    cam_cfgs[cid] = c

        containers = []

        # Build Container 1
        c1_images = self._build_container_images(c1_dict, cam_cfgs, is_rear_container=False)
        if not c1_images:
            raise ValueError(f"No valid proof images available for front container {cntr1}")

        conf1 = _safe_float(event_record.get('ocr_confidence', 0.95), 0.95)
        size1 = resolve_container_size(event_record.get('iso_code_1'), event_record.get('container_size'), is_twin=is_twin)
        containers.append({
            'ContainerNumber': cntr1,
            'ContainerNumberConfidence': round(conf1, 2),
            'Size': str(size1),
            'Position': 'FRONT' if is_twin else 'SINGLE',
            'Images': c1_images,
        })

        # Build Container 2 (if twin)
        if is_twin and cntr2:
            c2_images = self._build_container_images(c2_dict, cam_cfgs, is_rear_container=True)
            if not c2_images:
                # If rear-specific snapshots weren't explicitly separated, take remaining cameras
                c2_images = self._build_container_images(c1_dict, cam_cfgs, is_rear_container=True)
            if c2_images:
                conf2 = _safe_float(event_record.get('ocr_confidence_2', 0.95), 0.95)
                size2 = resolve_container_size(event_record.get('iso_code_2'), None, is_twin=True)
                containers.append({
                    'ContainerNumber': cntr2,
                    'ContainerNumberConfidence': round(conf2, 2),
                    'Size': str(size2),
                    'Position': 'REAR',
                    'Images': c2_images,
                })

        return {
            'id': event_id,
            'VisitId': visit_id,
            'Containers': containers,
        }

    def _build_container_images(
        self,
        best_snaps: Dict[str, Dict[str, Any]],
        cam_cfgs: Dict[str, Any],
        is_rear_container: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Assemble unique, non-colliding image objects for a container.
        Avoids CameraId duplicates within the container.
        Separates ImageTypes: approach camera -> FRONT, container proof -> REAR.
        """
        used_cameras = set()
        images = []

        # Sort cameras so container proof camera (e.g. gate-in-2 or gate-out-1) takes precedence
        def _sort_key(cam_id: str) -> int:
            c = cam_id.lower()
            if 'proof' in c or 'in-2' in c or 'out-1' in c or 'in-3' in c:
                return 0
            if 'in-1' in c or 'front' in c:
                return 1
            return 2

        sorted_cams = sorted(best_snaps.keys(), key=_sort_key)
        has_rear_type = False
        has_front_type = False

        for cam_id in sorted_cams:
            if cam_id in used_cameras:
                continue

            item = best_snaps[cam_id]
            snap = item['snap']
            jpeg_bytes = item['data']
            img_hash = item['hash']
            ts = snap.get('timestamp') or utcnow()

            # Determine distinct ImageType
            raw_type = resolve_camera_image_type(cam_id, cam_cfgs)
            if is_rear_container:
                img_type = 'REAR'
            else:
                # For container 1, if approach camera is FRONT and proof camera is LEFT/REAR:
                if 'in-2' in cam_id.lower() or 'proof' in str(snap.get('image_path', '')).lower():
                    img_type = 'REAR' if has_front_type else 'FRONT'
                elif 'in-1' in cam_id.lower():
                    img_type = 'FRONT' if not has_front_type else 'REAR'
                else:
                    img_type = raw_type

            if img_type == 'FRONT':
                has_front_type = True
            elif img_type == 'REAR':
                has_rear_type = True

            b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
            images.append({
                'ImageType': img_type,
                'CameraId': cam_id,
                'CapturedAt': _iso_utc(ts),
                'Image': f"data:image/jpeg;base64,{b64_str}",
                '_image_path': snap.get('image_path', ''),
                '_image_hash': img_hash,
            })
            used_cameras.add(cam_id)

        return images

    def _record_payload_proofs(
        self,
        event_id: str,
        cloud_event_id: str,
        payload: Dict[str, Any],
        proof_version: int,
    ):
        """Record all images in the payload into acknowledged_proofs in SQLite."""
        if not hasattr(self.repository, 'record_acknowledged_proof'):
            return

        for container in payload.get('Containers', []):
            c_num = container.get('ContainerNumber')
            if not c_num:
                continue
            for img in container.get('Images', []):
                cam_id = img.get('CameraId')
                if not cam_id:
                    continue
                self.repository.record_acknowledged_proof(
                    event_id=event_id,
                    cloud_event_id=cloud_event_id,
                    container_number=c_num,
                    camera_id=cam_id,
                    image_path=img.get('_image_path', ''),
                    image_hash=img.get('_image_hash', ''),
                    capture_timestamp=_parse_ts(img.get('CapturedAt')),
                    proof_version=proof_version,
                )


ProofUpdater = CloudProofUpgrader
