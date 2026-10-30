from __future__ import annotations
"""Motorized lens controller for Dahua / CP Plus IP cameras with motorized varifocal lenses.

Supports optical zoom (Tele/Wide), focus adjustment (Far/Near), one-click AutoFocus,
and anti-motion-blur night/day exposure, 3D denoise, sharpness, illuminator, and encoding tuning.
Communicates directly with the camera's secure HTTPS JSON-RPC endpoint (/cpapi2_Login and /cpapi2).
"""
import hashlib
from http.cookiejar import CookieJar
import json
import logging
import ssl
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger('gate')


class DahuaLensClient:
    def __init__(self, host: str, username: str = "admin", password: str = "Prosper@123"):
        self.host = host
        self.username = username
        self.password = password
        self.base_url = f"https://{host}"
        self.session = None
        self.vi_object = None
        self._lock = threading.RLock()
        self.ctx = ssl._create_unverified_context()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.ctx),
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self._req_id = 1

    def _rpc(self, path: str, payload: dict, timeout: float = 3.0) -> dict:
        body = json.dumps(payload)
        etag = hashlib.sha256(body.encode('utf-8')).hexdigest()
        headers = {
            'Content-Type': 'application/json',
            'ETag': etag,
            'X-Requested-With': 'XMLHttpRequest'
        }
        if self.session:
            headers['Cookie'] = f'DWebClientSessionID={self.session}; WebClientSessionID={self.session}; username={self.username}'
        req = urllib.request.Request(f"{self.base_url}{path}", data=body.encode('utf-8'), headers=headers)
        with self.opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def login(self) -> bool:
        with self._lock:
            try:
                self.session = None
                # Step 1: Challenge
                self._req_id += 1
                res1 = self._rpc('/cpapi2_Login', {
                    "method": "user.signin",
                    "params": {"userName": self.username, "password": "", "clientType": "Web3.0"},
                    "id": self._req_id
                })
                challenge_sess = res1.get('session')
                params1 = res1.get('params', {})
                realm = params1.get('realm', '')
                random = params1.get('random', '')
                if not realm or not random:
                    log.warning("dahua_lens_login_failed", extra={"reason": "missing_realm_or_random"})
                    return False

                # Step 2: Digest auth
                ha1 = hashlib.md5(f"{self.username}:{realm}:{self.password}".encode()).hexdigest().upper()
                pwd_hash = hashlib.md5(f"{self.username}:{random}:{ha1}".encode()).hexdigest().upper()
                self._req_id += 1
                res2 = self._rpc('/cpapi2_Login', {
                    "method": "user.signin",
                    "params": {
                        "userName": self.username,
                        "password": pwd_hash,
                        "clientType": "Web3.0",
                        "authorityType": params1.get("encryption", "Default"),
                        "passwordType": "Default",
                    },
                    "id": self._req_id,
                    "session": challenge_sess
                })
                if not res2.get('result'):
                    log.warning("dahua_lens_login_rejected", extra={"host": self.host})
                    return False
                self.session = res2.get('session', challenge_sess)

                # Step 3: devVideoInput instance
                self._req_id += 1
                vi = self._rpc('/cpapi2', {
                    "method": "devVideoInput.factory.instance",
                    "params": {"channel": 0},
                    "id": self._req_id,
                    "session": self.session
                })
                self.vi_object = vi.get('result')
                return self.vi_object is not None
            except Exception as e:
                log.error("dahua_lens_connection_error", extra={"error": str(e), "host": self.host})
                return False

    def _ensure_connected(self) -> bool:
        if self.session and self.vi_object is not None:
            return True
        return self.login()

    def get_status(self) -> dict:
        """Fetch current motorized lens position, zoom, and focus steps."""
        with self._lock:
            if not self._ensure_connected():
                return {"supported": False, "error": "Camera offline or credentials rejected"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.getFocusStatus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                if not res.get('result'):
                    # Session may have expired, retry once
                    self.session = None
                    if not self.login():
                        return {"supported": False, "error": "Session expired"}
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.getFocusStatus",
                        "params": None,
                        "id": self._req_id,
                        "session": self.session,
                        "object": self.vi_object
                    })
                status = res.get('params', {}).get('status', {})
                return {
                    "supported": True,
                    "status": status.get('Status', 'Normal'),
                    "zoom": status.get('Zoom', 0.0),
                    "focus": status.get('Focus', 0.0),
                    "zoom_steps": status.get('ZoomMotorSteps', 0),
                    "focus_steps": status.get('FocusMotorSteps', 0),
                }
            except Exception as e:
                return {"supported": False, "error": str(e)}

    def autofocus(self) -> dict:
        """Trigger camera optical auto-focus routine."""
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.autoFocus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result'))}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def stop_autofocus(self) -> dict:
        """Stop auto-focus routine."""
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.stopAutoFocus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result'))}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def zoom(self, action: str) -> dict:
        """
        action: 'in' | 'out' | 'stop' | 'step_in' | 'step_out'
        """
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                if action == 'step_in':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0.05},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.35)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'step_out':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": -0.05},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.35)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'in':
                    zoom_val = 0.05
                elif action == 'out':
                    zoom_val = -0.05
                else:  # 'stop'
                    zoom_val = 0

                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.adjustFocusContinuously",
                    "params": {"focus": -1, "zoom": zoom_val},
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result')), "action": action}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def focus(self, action: str) -> dict:
        """
        action: 'far' | 'near' | 'stop' | 'step_far' | 'step_near'
        """
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                if action == 'step_far':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0.03, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.3)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'step_near':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -0.03, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.3)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'far':
                    focus_val = 0.03
                elif action == 'near':
                    focus_val = -0.03
                else:  # 'stop'
                    focus_val = 0

                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.adjustFocusContinuously",
                    "params": {"focus": focus_val, "zoom": -1},
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result')), "action": action}
            except Exception as e:
                return {"success": False, "error": str(e)}

    SHUTTER_MAP = {
        '1/1000s': 1,
        '1/1000': 1,
        '1/500s': 2,
        '1/500': 2,
        '1/250s': 4,
        '1/250': 4,
        '1/100s': 10,
        '1/100': 10,
        '1/50s': 20,
        '1/50': 20,
        '1/25s': 40,
        '1/25': 40,
        'auto': 40,
    }
    INV_SHUTTER_MAP = {
        1: '1/1000s',
        2: '1/500s',
        4: '1/250s',
        10: '1/100s',
        20: '1/50s',
        40: '1/25s',
    }

    def get_exposure(self) -> dict:
        """Fetch current exposure mode and shutter speed configuration across Day/Night/Normal profiles."""
        with self._lock:
            if not self._ensure_connected():
                return {"supported": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInExposure"},
                    "id": self._req_id,
                    "session": self.session
                })
                if not res.get('result'):
                    return {"supported": False, "error": "Failed to get exposure config"}
                outer = res.get('params', {}).get('table', [])
                table = outer[0] if outer and isinstance(outer[0], list) else outer
                if not table:
                    return {"supported": False, "error": "Empty exposure config"}
                day_prof = table[0]
                night_prof = table[1] if len(table) > 1 else day_prof
                val2 = night_prof.get('Value2', day_prof.get('Value2', 40))
                mode_num = night_prof.get('Mode', day_prof.get('Mode', 0))
                shutter_label = self.INV_SHUTTER_MAP.get(val2, f"{val2}ms")
                return {
                    "supported": True,
                    "mode": "manual" if mode_num == 4 else ("custom_range" if mode_num == 8 else "auto"),
                    "shutter_speed": shutter_label,
                    "shutter_ms": val2,
                    "day_shutter_ms": day_prof.get('Value2', 40),
                    "night_shutter_ms": night_prof.get('Value2', 40),
                    "gain_max": night_prof.get('GainMax', 50),
                    "compensation": night_prof.get('Compensation', 50)
                }
            except Exception as e:
                return {"supported": False, "error": str(e)}

    def set_shutter_speed(self, shutter: str, mode: str = "auto") -> dict:
        """
        Configure electronic shutter speed and exposure mode across Day, Night, and Normal profiles.
        shutter: '1/1000s' | '1/500s' | '1/250s' | '1/100s' | '1/50s' | '1/25s' | 'auto'
        mode: 'auto' (custom shutter ceiling Mode=8 on CP Plus) | 'manual' (fixed shutter Mode=4)
        """
        shutter_key = shutter.lower()
        ms = self.SHUTTER_MAP.get(shutter_key)
        if ms is None:
            return {"success": False, "error": f"Invalid shutter speed '{shutter}'. Supported: {list(self.INV_SHUTTER_MAP.values())}"}
        if shutter_key == "auto":
            mode_val = 0
        elif mode == "manual":
            mode_val = 4
        else:
            mode_val = 8  # CP Plus Custom Shutter Range (Value1..Value2)
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInExposure"},
                    "id": self._req_id,
                    "session": self.session
                })
                outer = res.get('params', {}).get('table', [])
                if not outer:
                    return {"success": False, "error": "Could not read exposure table"}
                profiles = outer[0] if isinstance(outer[0], list) else outer
                for profile in profiles:
                    profile['Mode'] = mode_val
                    profile['Value2'] = ms
                    profile['SlowShutter'] = False
                    profile['SlowAutoExposure'] = 0
                    if mode_val == 4:
                        profile['Value1'] = ms
                        profile['GainMax'] = max(int(profile.get('GainMax', 50)), 70)
                    elif mode_val == 8:
                        profile['Value1'] = 0.1 if ms <= 2 else 0.5
                        profile['GainMax'] = max(int(profile.get('GainMax', 50)), 70)
                    else:
                        profile['Value1'] = 0
                self._req_id += 1
                set_res = self._rpc('/cpapi2', {
                    "method": "configManager.setConfig",
                    "params": {"name": "VideoInExposure", "table": outer},
                    "id": self._req_id,
                    "session": self.session
                })
                ok = bool(set_res.get('result'))
                return {
                    "success": ok,
                    "shutter_speed": self.INV_SHUTTER_MAP.get(ms, f"{ms}ms"),
                    "shutter_ms": ms,
                    "mode": mode
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    def optimize_night_imaging(
        self,
        day_max_shutter_ms: float = 4.0,
        night_max_shutter_ms: float = 6.0,
        gain_max: int = 75,
        tnf_level: int = 20,
        snf_level: int = 45,
        sharpness: int = 65,
        light_power: int = 50,
        bitrate_kbps: int = 6144,
        gop: int = 25,
        color_mode: bool = True,
    ) -> dict:
        """Apply comprehensive anti-motion-blur settings for fast-moving containers at night via CP Plus /cpapi2."""
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "host": self.host, "error": "Camera offline or login failed"}
            applied: dict[str, bool] = {}
            try:
                # 0. VideoInDayNight (keep cameras in Full Color mode instead of switching to B&W IR when shutter is fast)
                self._req_id += 1
                dn_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInDayNight"},
                    "id": self._req_id,
                    "session": self.session
                })
                dn_table = dn_res.get("params", {}).get("table")
                if dn_table and len(dn_table) > 0:
                    for prof in dn_table[0]:
                        prof["Mode"] = "Color" if color_mode else "Brightness"
                        prof["Type"] = "Electron" if color_mode else "Mechanism"
                    self._req_id += 1
                    r_dn = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "VideoInDayNight", "table": dn_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["VideoInDayNight"] = bool(r_dn.get("result"))

                # 1. VideoInExposure (Day=0, Night=1, Normal=2)
                self._req_id += 1
                exp_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInExposure"},
                    "id": self._req_id,
                    "session": self.session
                })
                exp_table = exp_res.get("params", {}).get("table")
                if exp_table and len(exp_table) > 0:
                    profiles = exp_table[0]
                    for idx, p in enumerate(profiles):
                        max_ms = day_max_shutter_ms if idx == 0 else night_max_shutter_ms
                        min_ms = 0.1 if idx == 0 else (0.5 if idx == 1 else 0.2)
                        p["Mode"] = 8
                        p["Value1"] = min_ms
                        p["Value2"] = max_ms
                        p["GainMin"] = 0
                        p["GainMax"] = gain_max if idx == 1 else max(60, gain_max - 5)
                        p["SlowShutter"] = False
                        p["SlowAutoExposure"] = 0
                    self._req_id += 1
                    r = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "VideoInExposure", "table": exp_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["VideoInExposure"] = bool(r.get("result"))

                # 2. VideoInDenoise (lower temporal TnfLevel to prevent ghosting trails)
                self._req_id += 1
                den_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInDenoise"},
                    "id": self._req_id,
                    "session": self.session
                })
                den_table = den_res.get("params", {}).get("table")
                if den_table and len(den_table) > 0:
                    for p in den_table[0]:
                        p["2DEnable"] = True
                        p["2DLevel"] = 3
                        p["3DType"] = "Manual"
                        if isinstance(p.get("3DAutoType"), dict):
                            p["3DAutoType"]["AutoLevel"] = 25
                        if isinstance(p.get("3DManulType"), dict):
                            p["3DManulType"]["SnfLevel"] = snf_level
                            p["3DManulType"]["TnfLevel"] = tnf_level
                        if isinstance(p.get("DenoiseAlgorithm1"), dict):
                            p["DenoiseAlgorithm1"]["Type"] = "Manual"
                            p["DenoiseAlgorithm1"]["SnfLevel"] = snf_level
                            p["DenoiseAlgorithm1"]["TnfLevel"] = tnf_level
                    self._req_id += 1
                    r = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "VideoInDenoise", "table": den_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["VideoInDenoise"] = bool(r.get("result"))

                # 3. VideoInSharpness
                self._req_id += 1
                shp_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInSharpness"},
                    "id": self._req_id,
                    "session": self.session
                })
                shp_table = shp_res.get("params", {}).get("table")
                if shp_table and len(shp_table) > 0:
                    for p in shp_table[0]:
                        p["Mode"] = 1
                        p["Sharpness"] = sharpness
                        p["Level"] = sharpness
                    self._req_id += 1
                    r = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "VideoInSharpness", "table": shp_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["VideoInSharpness"] = bool(r.get("result"))

                # 4. Lighting_V2 (boost IR / WhiteLight intensity for fast night shutter)
                self._req_id += 1
                lgt_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "Lighting_V2"},
                    "id": self._req_id,
                    "session": self.session
                })
                lgt_table = lgt_res.get("params", {}).get("table")
                if lgt_table and len(lgt_table) > 0:
                    for prof in lgt_table[0]:
                        for item in prof:
                            for nl in item.get("NearLight", []):
                                nl["Light"] = light_power
                            for fl in item.get("FarLight", []):
                                fl["Light"] = light_power
                    self._req_id += 1
                    r = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "Lighting_V2", "table": lgt_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["Lighting_V2"] = bool(r.get("result"))

                # 5. Encode (boost bitrate and shorten GOP for sharp motion edges)
                self._req_id += 1
                enc_res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "Encode"},
                    "id": self._req_id,
                    "session": self.session
                })
                enc_table = enc_res.get("params", {}).get("table")
                if enc_table and len(enc_table) > 0:
                    for mf in enc_table[0].get("MainFormat", []):
                        v = mf.get("Video", {})
                        if v:
                            v["BitRate"] = bitrate_kbps
                            v["LimitBitRate"] = bitrate_kbps
                            v["GOP"] = gop
                            v["AiGOP"] = gop
                            v["Quality"] = 6
                    self._req_id += 1
                    r = self._rpc('/cpapi2', {
                        "method": "configManager.setConfig",
                        "params": {"name": "Encode", "table": enc_table},
                        "id": self._req_id,
                        "session": self.session
                    })
                    applied["Encode"] = bool(r.get("result"))

                return {
                    "success": all(applied.values()) and len(applied) > 0,
                    "host": self.host,
                    "applied": applied,
                    "settings": {
                        "day_shutter_range_ms": [0.1, day_max_shutter_ms],
                        "night_shutter_range_ms": [0.5, night_max_shutter_ms],
                        "gain_max": gain_max,
                        "tnf_level": tnf_level,
                        "snf_level": snf_level,
                        "sharpness": sharpness,
                        "light_power": light_power,
                        "bitrate_kbps": bitrate_kbps,
                        "gop": gop,
                    },
                }
            except Exception as e:
                return {"success": False, "host": self.host, "applied": applied, "error": str(e)}


# Global cache of DahuaLensClients per host
_lens_clients: dict[str, DahuaLensClient] = {}
_cache_lock = threading.Lock()


def get_lens_client_for_camera(runtime, camera_id: str) -> DahuaLensClient | None:
    """Resolve camera IP/credentials and retrieve or initialize a DahuaLensClient."""
    worker = runtime.manager.workers.get(camera_id)
    if not worker or not worker._source:
        return None
    source = worker._source
    parsed = urllib.parse.urlsplit(source)
    host = parsed.hostname
    if not host:
        return None

    username = urllib.parse.unquote(parsed.username) if parsed.username else (runtime.settings.camera_onvif_username or "admin")
    password = urllib.parse.unquote(parsed.password) if parsed.password else None
    if not password:
        pw_secret = runtime.settings.camera_onvif_password.get_secret_value()
        password = pw_secret or "Prosper@123"

    with _cache_lock:
        client = _lens_clients.get(host)
        if client is None or client.password != password:
            client = DahuaLensClient(host, username=username, password=password)
            _lens_clients[host] = client
        return client
