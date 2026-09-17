"""
========================================================================================
                  POE CAMERA DETECTIVE & LIVE RTSP UNLOCKER
========================================================================================
All-in-one tool to discover, authenticate, and test video streams for ANY camera vendor:
- CP Plus / Dahua
- Hikvision / Ezviz
- Uniview (UNV)
- Xiongmai (XM) / Generic ONVIF IP Cameras
========================================================================================
Usage:
    python tools/rtsp_tester/find_cameras.py
    python tools/rtsp_tester/find_cameras.py <password>
    (Example: python tools/rtsp_tester/find_cameras.py 123456)
========================================================================================
"""

import socket
import struct
import subprocess
import threading
import time
import json
import os
import sys
import re
import cv2
from typing import List, Dict, Any, Tuple

# Enable low-latency OpenCV FFMPEG
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32|stimeout;2500000"


# -----------------------------------------------------------------------------
# 1. NETWORK & ADAPTER DIAGNOSTICS
# -----------------------------------------------------------------------------
def inspect_ethernet_adapter():
    print("\n" + "=" * 75)
    print("📡 [1/4] Inspecting Ethernet Adapter & Subnet Routing...")
    print("=" * 75)
    try:
        cmd = 'powershell "Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias *Ethernet* | Select-Object IPAddress, InterfaceAlias, PrefixLength"'
        out = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
        print("  Current Laptop Ethernet IP(s):")
        for line in out.strip().splitlines():
            if line.strip() and not line.startswith("IPAddress") and not line.startswith("---------"):
                print(f"    • {line.strip()}")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 2. MULTI-VENDOR DISCOVERY (SADP, ConfigTool, ONVIF, NetIP, Uniview)
# -----------------------------------------------------------------------------
def get_vendor_probes() -> List[Tuple[bytes, str, int, str]]:
    probes = []

    # CP Plus / Dahua JSON ConfigTool
    cp_json = json.dumps({"method": "client.search", "params": {"mac": ""}}).encode("utf-8")
    probes.append((cp_json, "255.255.255.255", 37810, "CP Plus / Dahua"))
    probes.append((cp_json, "239.255.255.250", 37810, "CP Plus / Dahua"))

    # CP Plus / Dahua Binary
    dh_bin = b"\xa0\x00\x00\x60\x00\x00\x00\x00DHIP\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    probes.append((dh_bin, "255.255.255.255", 37810, "CP Plus Binary"))
    probes.append((dh_bin, "255.255.255.255", 37777, "CP Plus Private"))

    # Hikvision SADP
    hik_xml = b"<?xml version=\"1.0\" encoding=\"utf-8\"?><Probe><Types>inquiry</Types></Probe>"
    probes.append((hik_xml, "239.255.255.250", 37020, "Hikvision SADP"))
    probes.append((hik_xml, "255.255.255.255", 37020, "Hikvision SADP"))

    # Xiongmai / NetIP
    xm_bin = b"\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfa\x05\x00\x00\x00\x00\x00\x00"
    probes.append((xm_bin, "255.255.255.255", 34567, "Xiongmai / XM"))

    # ONVIF WS-Discovery
    onvif_msg = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Envelope xmlns:dn="http://www.onvif.org/ver10/network/wsdl" xmlns="http://www.w3.org/2003/05/soap-envelope">'
        '<Header><wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">uuid:c898b350-0a17-48f8-a1c2-3e2c347df277</wsa:MessageID>'
        '<wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>'
        '<wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action></Header>'
        '<Body><Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery"><Types>dn:NetworkVideoTransmitter</Types></Probe></Body>'
        '</Envelope>'
    ).encode("utf-8")
    probes.append((onvif_msg, "239.255.255.250", 3702, "ONVIF Multicast"))
    probes.append((onvif_msg, "255.255.255.255", 3702, "ONVIF Broadcast"))

    return probes


def send_discovery_probes(timeout_sec: float = 2.5) -> List[Dict[str, Any]]:
    print("\n" + "=" * 75)
    print("📡 [2/4] Broadcasting Multi-Vendor Probes (CP Plus, Hikvision, ONVIF, XM)...")
    print("=" * 75)
    found = []
    seen = set()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.3)

    for payload, host, port, desc in get_vendor_probes():
        try:
            sock.sendto(payload, (host, port))
        except Exception:
            pass

    t_end = time.monotonic() + timeout_sec
    while time.monotonic() < t_end:
        try:
            data, addr = sock.recvfrom(4096)
            ip = addr[0]
            if ip.startswith("127."):
                continue

            raw = data.decode("utf-8", errors="ignore")
            cam_info = {"ip": ip, "vendor": "IP Camera", "mac": "", "model": ""}

            # CP Plus / Dahua parser
            try:
                js = json.loads(raw)
                if "params" in js:
                    p = js["params"]
                    cam_info["ip"] = p.get("IPv4Address", {}).get("IPAddress", ip)
                    cam_info["vendor"] = "CP Plus / Dahua"
                    cam_info["mac"] = p.get("MAC", "")
                    cam_info["model"] = p.get("DeviceType", "")
            except Exception:
                pass

            # Hikvision parser
            if "<ProbeMatch>" in raw or "Hikvision" in raw or "HIKVISION" in raw:
                cam_info["vendor"] = "Hikvision"
                ip_m = re.search(r"<IPv4Address>([\d\.]+)</IPv4Address>", raw)
                if ip_m:
                    cam_info["ip"] = ip_m.group(1)
                mac_m = re.search(r"<MAC>([^<]+)</MAC>", raw)
                if mac_m:
                    cam_info["mac"] = mac_m.group(1)
                mod_m = re.search(r"<DeviceDescription>([^<]+)</DeviceDescription>", raw)
                if mod_m:
                    cam_info["model"] = mod_m.group(1)

            # ONVIF parser
            elif "schemas.xmlsoap.org" in raw or "onvif" in raw.lower():
                cam_info["vendor"] = "ONVIF Standard"
                ip_m = re.search(r"http://([\d\.]+):?(\d+)?/", raw)
                if ip_m:
                    cam_info["ip"] = ip_m.group(1)

            if cam_info["ip"] not in seen:
                seen.add(cam_info["ip"])
                found.append(cam_info)
                print(f"  🎯 Discovered Broadcast: {cam_info['ip']} ({cam_info['vendor']} {cam_info['model']})")
        except socket.timeout:
            pass
        except Exception:
            pass

    sock.close()
    return found


# -----------------------------------------------------------------------------
# 3. FAST TCP PORT & ARP SWEEP ACROSS MULTIPLE SUBNETS
# -----------------------------------------------------------------------------
def fast_subnet_port_sweep() -> List[Dict[str, Any]]:
    print("\n" + "=" * 75)
    print("🔍 [3/4] Sweeping Ports 554, 8000, 37777 Across 192.168.1.x & 192.168.0.x...")
    print("=" * 75)

    candidate_ips = []
    # 1. ARP table
    try:
        arp_out = subprocess.check_output("arp -a", shell=True).decode("utf-8", errors="ignore")
        for match in re.finditer(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", arp_out):
            ip = match.group(1)
            if not ip.endswith(".255") and not ip.startswith("224.") and not ip.startswith("239.") and not ip.startswith("127."):
                if ip not in candidate_ips:
                    candidate_ips.append(ip)
    except Exception:
        pass

    # 2. Known default camera IPs
    defaults = [
        "192.168.1.4", "192.168.1.155", "192.168.0.10", "192.168.0.11",
        "192.168.1.64", "192.168.1.108", "192.168.1.250", "192.168.0.64",
        "192.168.0.108", "192.168.0.250", "192.168.1.10", "192.168.1.13"
    ]
    for d in defaults:
        if d not in candidate_ips:
            candidate_ips.append(d)

    # 3. Add entire /24 for 192.168.1 and 192.168.0
    for sub in ["192.168.1", "192.168.0"]:
        for i in range(1, 255):
            ip_str = f"{sub}.{i}"
            if ip_str not in candidate_ips:
                candidate_ips.append(ip_str)

    found = []
    lock = threading.Lock()

    def _probe_ip(ip):
        for port in [554, 8000, 37777, 8888, 34567, 80]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.20)
                res = s.connect_ex((ip, port))
                s.close()
                if res == 0:
                    with lock:
                        if not any(f["ip"] == ip for f in found):
                            vendor = "Generic Camera"
                            if port == 554:
                                vendor = "RTSP Open"
                            elif port == 8000:
                                vendor = "Hikvision Device"
                            elif port == 37777:
                                vendor = "CP Plus / Dahua Device"
                            found.append({"ip": ip, "vendor": vendor, "port": port})
                    break
            except Exception:
                pass

    for i in range(0, len(candidate_ips), 64):
        batch = candidate_ips[i:i+64]
        threads = []
        for ip in batch:
            t = threading.Thread(target=_probe_ip, args=(ip,), daemon=True)
            batch_threads = threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=0.35)

    return found


# -----------------------------------------------------------------------------
# 4. LIVE RTSP VIDEO STREAM AUTHENTICATION & VERIFICATION
# -----------------------------------------------------------------------------
def verify_and_unlock_streams(camera_ips: List[str], custom_password: str = None) -> Dict[str, Dict[str, Any]]:
    print("\n" + "=" * 75)
    print("🎥 [4/4] Verifying Live Video Streams & Frame Capture...")
    print("=" * 75)

    passwords = []
    if custom_password:
        passwords.append(custom_password)
    for p in ["Admin@123", "Prosper@123"]:
        if p not in passwords:
            passwords.append(p)

    users = ["admin"]

    paths = [
        "/cam/realmonitor?channel=1&subtype=1",
        "/cam/realmonitor?channel=1&subtype=0",
        "/Streaming/Channels/102",
        "/Streaming/Channels/101",
        "/h264/ch1/sub/av_stream",
        "/h264/ch1/main/av_stream",
        "/unicast/c1/s1/live",
        "/live",
        "/onvif1"
    ]

    results = {}

    for ip in camera_ips:
        print(f"\n▶ Testing IP: {ip} ...")
        unlocked = False

        # Fast socket check
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            r = s.connect_ex((ip, 554))
            s.close()
            if r != 0:
                print(f"  ⚠️  Cannot reach port 554 on {ip} directly. (Laptop needs dual-subnet IP).")
                continue
        except Exception:
            continue

        for u in users:
            if unlocked:
                break
            for p in passwords:
                if unlocked:
                    break
                for path in paths:
                    url = f"rtsp://{u}:{p}@{ip}:554{path}"
                    try:
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                h, w = frame.shape[:2]
                                print("\n" + "*" * 65)
                                print(f"  🎉 CAMERA LIVE & CAPTURING FRAMES!")
                                print(f"     ▶ IP:         {ip}")
                                print(f"     ▶ Credentials: {u}:{p}")
                                print(f"     ▶ Resolution: {w}x{h} px")
                                print(f"     ▶ RTSP URL:   {url}")
                                print("*" * 65)
                                results[ip] = {
                                    "url": url,
                                    "user": u,
                                    "password": p,
                                    "res": f"{w}x{h}"
                                }
                                unlocked = True
                                cap.release()
                                break
                            cap.release()
                    except Exception:
                        pass

        if not unlocked:
            print(f"  ❌ {ip}: Port 554 is reachable, but password was not matched. Provide custom password as argument.")

    return results


# -----------------------------------------------------------------------------
# MAIN RUNNER
# -----------------------------------------------------------------------------
def main():
    custom_pw = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "#" * 75)
    print("          POE CAMERA AUTO-DETECTIVE & ZERO-LAG RTSP UNLOCKER")
    print("#" * 75)
    if custom_pw:
        print(f"  🔑 Using Custom Password Priority: '{custom_pw}'")

    # Step 1: Adapter check
    inspect_ethernet_adapter()

    # Step 2: Discovery probes
    broadcast_cams = send_discovery_probes(timeout_sec=2.0)

    # Step 3: Subnet sweep
    sweep_cams = fast_subnet_port_sweep()

    # Merge IP list
    unique_cams = {}
    for c in broadcast_cams + sweep_cams:
        ip = c["ip"]
        if ip not in unique_cams:
            unique_cams[ip] = c

    print("\n" + "=" * 75)
    print(f"📋 ALL DETECTED HARDWARE CAMERAS ({len(unique_cams)} Found):")
    print("=" * 75)
    for i, (ip, info) in enumerate(unique_cams.items(), 1):
        v = info.get("vendor", "Camera")
        m = info.get("model", "")
        mac = info.get("mac", "")
        m_str = f" | Model: {m}" if m else ""
        mac_str = f" | MAC: {mac}" if mac else ""
        print(f"  [{i}] IP: {ip:<16} | Type: {v}{m_str}{mac_str}")

    # Step 4: Stream verification
    working_streams = verify_and_unlock_streams(list(unique_cams.keys()), custom_pw)

    print("\n" + "#" * 75)
    print("                     FINAL READY-TO-USE RTSP URLS")
    print("#" * 75)
    if working_streams:
        for ip, data in working_streams.items():
            print(f"\n  ✅ {ip} ({data['res']})")
            print(f"     URL: {data['url']}")
    else:
        print("\n  ⚠️  No active video streams were unlocked yet.")
        print("\n  TROUBLESHOOTING:")
        print("  1. If camera IPs are in 192.168.0.x, ensure laptop Ethernet has 192.168.0.100 assigned.")
        print("  2. Pass your camera password directly:")
        print("     python tools/rtsp_tester/find_cameras.py YourPasswordHere")
    print("\n" + "#" * 75 + "\n")


if __name__ == "__main__":
    main()
