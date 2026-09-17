"""
Prosper Smart Yard Cloud Outbox Dispatcher
Asynchronously synchronizes confirmed GateVision events and evidence images
to the remote Prosper Smart Yard API (syapi.prosperassettracking.com).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import logging
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Dict, Optional
import uuid

import cv2
import requests

log = logging.getLogger('gate.cloud')


def _iso_utc(dt: Any = None) -> str:
    if dt is None:
        d = datetime.now(timezone.utc)
    elif isinstance(dt, (int, float)):
        try:
            d = datetime.fromtimestamp(dt, tz=timezone.utc)
        except Exception:
            d = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        return dt
    elif isinstance(dt, datetime):
        if dt.tzinfo is None:
            d = dt.replace(tzinfo=timezone.utc)
        else:
            d = dt.astimezone(timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


class ProsperSession:
    """Manages authentication tokens and HTTP requests to Prosper Smart Yard."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.prosper_api_base_url.rstrip('/')
        self.site_id = (settings.prosper_site_id or '').strip()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()

    def get_auth_headers(self) -> Dict[str, str]:
        """Return Authorization or API key headers."""
        if self.settings.prosper_bearer_token:
            return {'Authorization': f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'}
        if self.settings.prosper_api_key:
            return {'x-api-key': self.settings.prosper_api_key.get_secret_value()}

        # Auto-login credential flow
        with self._lock:
            now = time.monotonic()
            if self._token and now < self._token_expires_at - 30:
                return {'Authorization': f'Bearer {self._token}'}

            username = getattr(self.settings, 'prosper_username', None) or self.settings.prosper_email or 'admin'
            configured_pw = self.settings.prosper_password.get_secret_value() if self.settings.prosper_password else ''
            passwords_to_try = [configured_pw] if configured_pw else []
            for candidate in ['Admin@123', 'Prosper@123']:
                if candidate not in passwords_to_try:
                    passwords_to_try.append(candidate)

            try:
                login_url = f'{self.base_url}/api/auth/login'
                res = None
                for pw in passwords_to_try:
                    payload = {
                        'UserName': username,
                        'Password': pw,
                    }
                    if self.settings.prosper_site_code:
                        payload['siteCode'] = self.settings.prosper_site_code

                    res = self.session.post(login_url, json=payload, timeout=10)
                    if res.status_code not in (200, 201):
                        legacy_payload = {
                            'siteCode': self.settings.prosper_site_code or '',
                            'email': self.settings.prosper_email or username,
                            'password': pw,
                        }
                        res = self.session.post(login_url, json=legacy_payload, timeout=10)

                    if res.status_code in (200, 201):
                        data = res.json()
                        self._token = data.get('accessToken') or data.get('token')
                        expires_in = data.get('expiresIn', 3600)
                        self._token_expires_at = time.monotonic() + float(expires_in)
                        log.info('prosper_auth_token_refreshed', extra={'username': username})
                        return {'Authorization': f'Bearer {self._token}'}

                if res is not None:
                    log.warning('prosper_auth_login_failed', extra={'status_code': res.status_code, 'body': res.text[:200]})
            except Exception as e:
                log.warning('prosper_auth_login_error', extra={'error': str(e)})

        return {}

    def upload_image(self, image_data: bytes | Any, trailer_number: str = '', camera_id: str = '',
                     device_id: str = '') -> Optional[str]:
        """Upload snapshot to Prosper and return public/stored URL."""
        if not self.site_id:
            return None

        if not isinstance(image_data, bytes):
            ok, buf = cv2.imencode('.jpg', image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None
            image_data = buf.tobytes()

        headers = self.get_auth_headers()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        base64_uri = f'data:image/jpeg;base64,{base64_str}'

        # 1. Try S3 Base64 upload endpoint (used in previous codebase)
        s3_url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers/upload-s3'
        s3_payload = {
            'trailerNumber': trailer_number or 'UNKNOWN',
            'scac': trailer_number[:4] if len(trailer_number) >= 4 and trailer_number[:4].isalpha() else 'UNKN',
            'deviceId': device_id,
            'cameraId': camera_id,
            'imagePath': base64_uri,
        }
        try:
            res = self.session.post(s3_url, json=s3_payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                nested = data.get('data') if isinstance(data.get('data'), dict) else {}
                img_id = data.get('id') or nested.get('id') or nested.get('imageMetadataId') or data.get('imageMetadataId')
                img_url = nested.get('fileUrl') or nested.get('imagePath') or data.get('fileUrl') or data.get('imagePath') or data.get('url')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_s3_uploaded', extra={'img_id': img_id, 'url': img_url})
                return img_url
        except Exception as e:
            log.warning('prosper_image_s3_upload_error', extra={'error': str(e)})

        # 2. Fallback to multipart upload
        url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers'
        files = {'file': ('evidence.jpg', io.BytesIO(image_data), 'image/jpeg')}
        try:
            res = self.session.post(url, files=files, headers=headers, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                img_id = data.get('id') or data.get('imageMetadataId')
                img_url = data.get('url') or data.get('imageUrl') or data.get('fileUrl')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_uploaded', extra={'url': img_url})
                return img_url
            log.warning('prosper_image_upload_failed', extra={'status_code': res.status_code})
        except Exception as e:
            log.warning('prosper_image_upload_error', extra={'error': str(e)})

        return None

    def post_gate_event(self, payload: Dict[str, Any]) -> bool:
        """Send CreateGateEventRequest to Prosper."""
        if not self.site_id:
            log.warning('prosper_sync_skipped_missing_site_id')
            return False

        url = f'{self.base_url}/api/sites/{self.site_id}/gate-events'
        headers = self.get_auth_headers()
        headers['Content-Type'] = 'application/json'

        res = self.session.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code in (200, 201):
            log.info('prosper_gate_event_synced', extra={'event_id': payload.get('id')})
            return True
        log.warning('prosper_gate_event_rejected', extra={'status_code': res.status_code, 'body': res.text[:200]})
        return False


class CloudOutboxDispatcher:
    """
    Dedicated non-blocking worker thread that serializes confirmed gate events,
    uploads snapshot evidence images, and dispatches gate events to Prosper Smart Yard.
    """

    def __init__(self, settings, metrics=None):
        self.settings = settings
        self.metrics = metrics
        self.enabled = bool(settings.prosper_enabled)
        self.session = ProsperSession(settings) if self.enabled else None
        self.queue: Queue = Queue(maxsize=settings.prosper_queue_size)
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Deterministic device UUID fallback
        self.device_uuid = (
            settings.prosper_device_uuid
            or str(uuid.uuid5(uuid.NAMESPACE_DNS, 'gatevision.edge.local'))
        )

    def start(self):
        if not self.enabled:
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name='prosper-dispatcher', daemon=True)
        self._worker_thread.start()
        log.info('prosper_cloud_dispatcher_started')

    def stop(self, timeout: float = 2.0):
        if not self.enabled:
            return
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        log.info('prosper_cloud_dispatcher_stopped')

    def enqueue(self, obs: Any, event_id: str, image_bytes: Optional[bytes] = None):
        """Enqueue confirmed gate event for background dispatch."""
        if not self.enabled:
            return
        if not obs.confirmed:
            return

        item = {
            'event_id': event_id,
            'obs': obs,
            'image_bytes': image_bytes if image_bytes is not None else obs.image,
            'enqueued_at': time.monotonic(),
            'retries': 0,
        }

        try:
            self.queue.put_nowait(item)
            if self.metrics:
                self.metrics.increment('prosper_events_queued')
        except Full:
            log.warning('prosper_cloud_queue_full_event_dropped', extra={'event_id': event_id})
            if self.metrics:
                self.metrics.increment('prosper_queue_dropped')

    def _run_loop(self):
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue

            event_id = item['event_id']
            obs = item['obs']
            img = item['image_bytes']

            success = False
            try:
                # 1. Upload snapshot evidence image first
                image_url = None
                cntr = obs.ocr.normalized_text if obs.ocr else ''
                if img is not None:
                    image_url = self.session.upload_image(
                        img,
                        trailer_number=cntr,
                        camera_id=obs.detection.camera_id,
                        device_id=self.device_uuid,
                    )
                img_id = getattr(self.session, 'last_image_metadata_id', None)

                # 2. Build CreateGateEventRequest payload
                payload = {
                    'id': event_id,
                    'gateId': obs.gate_id,
                    'timestamp': _iso_utc(obs.detection.frame_timestamp),
                    'containerNumber': cntr or None,
                    'trailerNumber': cntr or None,
                    'eventType': obs.direction if obs.direction in ('ENTRY', 'EXIT') else 'ENTRY',
                    'deviceId': self.device_uuid,
                    'confidence': float(obs.detection.confidence),
                }
                if img_id:
                    payload['imageMetadataId'] = img_id
                if image_url:
                    payload['imageUrl'] = image_url

                # 3. Post gate event
                success = self.session.post_gate_event(payload)

            except Exception as exc:
                log.warning('prosper_dispatch_exception', extra={'event_id': event_id, 'error': str(exc)})
                success = False

            if success:
                if self.metrics:
                    self.metrics.increment('prosper_events_synced')
                    self.metrics.latency('prosper_sync', time.monotonic() - item['enqueued_at'])
                self.queue.task_done()
            else:
                item['retries'] += 1
                if item['retries'] < self.settings.prosper_max_retries:
                    # Exponential backoff retry
                    backoff = min(2.0 ** item['retries'], 30.0)
                    self._stop.wait(backoff)
                    try:
                        self.queue.put_nowait(item)
                    except Full:
                        log.error('prosper_retry_dropped_queue_full', extra={'event_id': event_id})
                else:
                    log.error('prosper_event_sync_exhausted', extra={'event_id': event_id})
                    if self.metrics:
                        self.metrics.increment('prosper_sync_errors')
                self.queue.task_done()
