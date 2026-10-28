"""Persist the exact prospective upload, including its encoded JPEG bytes."""
import hashlib
import json
from pathlib import Path
import re
import threading
from datetime import datetime, timezone
import base64
from copy import deepcopy
from uuid import uuid4


class OutgoingPreview:
    _lock = threading.Lock()

    def __init__(self, root):
        self.root = Path(root) / 'outgoing-preview'

    def path(self, event_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', event_id):
            raise ValueError('Invalid event ID')
        return self.root / (event_id + '.json')

    def load(self, event_id):
        path = self.path(event_id)
        if not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding='utf-8'))
        payload = envelope['payload']
        digest = self.digest(payload)
        if digest != envelope['sha256']:
            raise ValueError('Saved upload preview failed integrity check')
        return payload

    @staticmethod
    def digest(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def freeze(self, event_id, payload):
        containers = payload.get('Containers') or []
        if not containers or any(not c.get('Images') for c in containers):
            raise ValueError('Upload preview has no complete image set')
        with self._lock:
            existing = self.load(event_id)
            if existing is not None:
                return existing
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path(event_id)
            temp = path.with_suffix('.' + uuid4().hex + '.tmp')
            try:
                temp.write_text(json.dumps({'sha256': self.digest(payload), 'payload': payload}), encoding='utf-8')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            return self.load(event_id)

    def revise_camera(self, event_id, camera_id, jpeg, captured_at, source_path, reason):
        """Explicit reviewed revision; automatic refreshes still cannot change proof.

        Preserve the prior envelope and change only this camera's encoded image.
        Caller must verify the source belongs to the visit and depicts its container.
        """
        if not jpeg.startswith(b'\xff\xd8') or not reason.strip():
            raise ValueError('An original JPEG and review reason are required')
        with self._lock:
            previous = self.load(event_id)
            if previous is None:
                raise ValueError('No frozen preview to revise')
            updated = deepcopy(previous)
            matches = [image for container in updated.get('Containers', [])
                       for image in container.get('Images', []) if image.get('CameraId') == camera_id]
            if len(matches) != 1:
                raise ValueError('Review one unambiguous container/camera proof at a time')
            revision = int(updated.get('ProofVersion') or 1) + 1
            image = matches[0]
            old_hash = image.get('_image_hash')
            image.update(Image='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(),
                         CapturedAt=captured_at, _image_path=source_path,
                         _image_hash=hashlib.sha256(jpeg).hexdigest())
            updated['ProofVersion'] = revision
            updated.setdefault('_local_proof_revisions', []).append({
                'camera_id': camera_id, 'reason': reason, 'previous_image_hash': old_hash,
                'source_path': source_path, 'revision': revision,
                'reviewed_at': datetime.now(timezone.utc).isoformat()})
            current = self.path(event_id)
            archive = self.root / f'{event_id}.before-v{revision}.json'
            if archive.exists():
                raise ValueError('Revision archive already exists')
            archive.write_bytes(current.read_bytes())
            temp = current.with_suffix('.' + uuid4().hex + '.tmp')
            try:
                temp.write_text(json.dumps({'sha256': self.digest(updated), 'payload': updated}), encoding='utf-8')
                temp.replace(current)
            finally:
                temp.unlink(missing_ok=True)
            return self.load(event_id)
