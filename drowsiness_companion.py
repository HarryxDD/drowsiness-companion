#!/usr/bin/env python3
import argparse
import collections
import json
import math
import os
import socket
import subprocess
import threading
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------- tunables
CALIBRATION_SEC = 3.0  # look at the camera with eyes open during this
EAR_RATIO = 0.70  # eyes "closed" when EAR < 70% of your open-eye EAR
EYE_SIGN_SEC = 1.2  # eyes closed this long = drowsiness sign (long blink)
EYE_ALARM_SEC = 2.5  # eyes closed this long = immediate alarm
MAR_YAWN = 0.55  # mouth aspect ratio above this = mouth wide open
YAWN_SEC = 1.0  # mouth open this long = yawn
NOD_DELTA = 0.07  # head pitched down this much past baseline = nodding
NOD_SEC = 0.8  # held this long = head nod
ESCALATE_WINDOW = 60.0  # another sign within this many s after a nudge -> alarm
BANNER_SEC = 4.0  # how long the Nudge / Alert stage stays lit

NUDGE_TEXT = "Hey, you look tired. Time for a break?"

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # p1..p6 for EAR
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS = (78, 308)
MOUTH_PAIRS = [(13, 14), (81, 178), (311, 402)]  # upper/lower inner lip
FOREHEAD, NOSE_TIP, CHIN = 10, 1, 152


# ---------------------------------------------------------------- geometry
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts, idx):
    p = [pts[i] for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-6)


def mouth_aspect_ratio(pts):
    width = dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]]) + 1e-6
    height = sum(dist(pts[a], pts[b]) for a, b in MOUTH_PAIRS) / len(MOUTH_PAIRS)
    return height / width


def head_pitch_ratio(pts):
    """Where the nose sits between forehead and chin (0..1). Grows when head drops forward."""
    top, nose, chin = pts[FOREHEAD][1], pts[NOSE_TIP][1], pts[CHIN][1]
    return (nose - top) / (chin - top + 1e-6)


# ---------------------------------------------------------------- optional Pi audio
def make_alarm_wav(path, seconds=2.0, rate=22050):
    t = np.arange(int(seconds * rate)) / rate
    freq = np.where((t * 4).astype(int) % 2 == 0, 880, 1320)
    on = (t * 8).astype(int) % 2 == 0
    samples = (0.9 * 32767 * np.sin(2 * np.pi * freq * t) * on).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


class PiAudio:
    def __init__(self):
        self.alarm_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alarm.wav"
        )
        if not os.path.exists(self.alarm_path):
            make_alarm_wav(self.alarm_path)
        self.proc = None

    def _run(self, cmd):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print(
                f"[pi-audio] '{cmd[0]}' not found - sudo apt install espeak-ng alsa-utils"
            )

    def say(self, text):
        self._run(["espeak-ng", "-s", "145", "-a", "170", text])

    def alarm(self):
        self._run(["aplay", "-q", self.alarm_path])


# ---------------------------------------------------------------- shared state
class Hub:
    """Latest video frame + an event stream, shared between detector and web clients."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.frame_id = None, 0
        self.seq = 0
        self.events = collections.deque(maxlen=400)
        self.logs = collections.deque(maxlen=40)
        self.recalibrate = threading.Event()

    def set_frame(self, jpeg):
        with self.cond:
            self.jpeg, self.frame_id = jpeg, self.frame_id + 1
            self.cond.notify_all()

    def wait_frame(self, last_id, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id, timeout)
            return self.frame_id, self.jpeg

    def push(self, ev):
        with self.cond:
            self.seq += 1
            ev["seq"] = self.seq
            self.events.append(ev)
            if ev["type"] == "log":
                self.logs.append(ev)
            self.cond.notify_all()

    def log(self, msg, level="info"):
        t = time.strftime("%H:%M:%S")
        print(t, msg, flush=True)
        self.push({"type": "log", "level": level, "msg": msg, "t": t})

    def snapshot(self):
        with self.cond:
            return self.seq, list(self.logs)

    def wait_events(self, after, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after, timeout)
            return [e for e in self.events if e["seq"] > after], self.seq


# ---------------------------------------------------------------- detector
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return MODEL_PATH
    print("Downloading face model (one time, ~4 MB)...", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)
    except Exception as e:
        raise SystemExit(
            f"Could not download the face model ({e}).\n"
            f"Connect the Pi to the internet once, or download\n  {MODEL_URL}\n"
            f"on the laptop and copy it next to this script as face_landmarker.task"
        )
    return MODEL_PATH


class FaceTracker:
    """MediaPipe Face Landmarker (Tasks API, works with mediapipe 0.10.x and 1.x)."""

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.last_ts = 0

    def landmarks(self, frame_bgr):
        """Returns a list of (x, y) pixel points, or None if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        res = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )
        if not res.face_landmarks:
            return None
        return [(p.x * w, p.y * h) for p in res.face_landmarks[0]]


class Detector:
    def __init__(self, hub, cam_index, pi_audio=None, tracker=None):
        self.hub, self.cam_index, self.pi_audio = hub, cam_index, pi_audio
        self.tracker = tracker or FaceTracker()
        self.cap = open_camera(cam_index)
        self.reset_calibration(time.time())
        self.closed_since = self.yawn_since = self.nod_since = None
        self.fired = dict(eye_sign=False, yawn=False, nod=False)
        self.last_nudge = self.last_alert = 0.0
        self.nudge_until = self.alert_until = 0.0
        self.nudges = self.alerts = 0

    def reset_calibration(self, now):
        self.calib_start, self.calib_ear, self.calib_pitch = now, [], []
        self.ear_thresh = self.open_ear = self.pitch_base = None

    # ---- stage 3 and 4
    def nudge(self, now, reason):
        self.nudges += 1
        self.last_nudge, self.nudge_until = now, now + BANNER_SEC
        self.hub.log(f"Nudge: {reason}", "nudge")
        self.hub.push({"type": "nudge", "text": NUDGE_TEXT, "reason": reason})
        if self.pi_audio:
            self.pi_audio.say(NUDGE_TEXT)

    def alert(self, now, reason):
        if now - self.last_alert < 2.5:  # an alarm is already sounding
            return
        self.last_alert = now
        self.alerts += 1
        self.alert_until = now + BANNER_SEC
        self.hub.log(f"ALERT: {reason}", "alert")
        self.hub.push({"type": "alert", "reason": reason})
        if self.pi_audio:
            self.pi_audio.alarm()

    def on_sign(self, now, reason):
        if self.last_nudge and now - self.last_nudge < ESCALATE_WINDOW:
            self.alert(now, f"{reason}, again after a nudge")
        else:
            self.nudge(now, reason)

    # ---- main loop
    def run(self):
        try:
            self._loop()
        except Exception as e:  # keep the web page alive and tell the user
            self.hub.log(f"Detector crashed: {e!r}", "alert")
            raise

    def _loop(self):
        hub = self.hub
        if self.cap.isOpened():
            hub.log("Camera started. Calibrating: look at the camera with eyes open")
        fails, last_metrics = 0, 0.0
        fps_t, fps_n, fps = time.time(), 0, 0.0

        while True:
            ok, frame = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                fails += 1
                if fails == 1:
                    hub.log("No camera image - check the USB camera", "warn")
                if fails % 10 == 0:
                    self.cap.release()
                    self.cap = open_camera(self.cam_index)
                    if self.cap.isOpened():
                        hub.log("Camera reconnected")
                time.sleep(0.3)
                continue
            fails = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            if hub.recalibrate.is_set():
                hub.recalibrate.clear()
                self.reset_calibration(now)
                hub.log("Recalibrating: look at the camera with eyes open")

            pts = self.tracker.landmarks(frame)
            face = pts is not None

            ear = mar = pitch = None
            eyes_closed = False
            if face:
                ear = (
                    eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)
                ) / 2
                mar = mouth_aspect_ratio(pts)
                pitch = head_pitch_ratio(pts)

                if self.ear_thresh is None:
                    self.calib_ear.append(ear)
                    self.calib_pitch.append(pitch)
                    if (
                        now - self.calib_start >= CALIBRATION_SEC
                        and len(self.calib_ear) > 10
                    ):
                        self.open_ear = float(np.median(self.calib_ear))
                        self.ear_thresh = self.open_ear * EAR_RATIO
                        self.pitch_base = float(np.median(self.calib_pitch))
                        hub.log(
                            f"Calibrated (open-eye EAR {self.open_ear:.2f}). Watching."
                        )
                else:
                    self._detect(now, ear, mar, pitch)
                    eyes_closed = self.closed_since is not None

                self._draw_face(frame, pts, eyes_closed, now)
            else:
                # face lost: stop running timers so they don't fire on stale data
                self.closed_since = self.yawn_since = self.nod_since = None

            if self.ear_thresh is None:
                left = max(0.0, CALIBRATION_SEC - (now - self.calib_start))
                txt = (
                    f"Calibrating... {left:.1f}s"
                    if face
                    else "Calibrating: no face in view"
                )
                cv2.putText(
                    frame,
                    txt,
                    (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                hub.set_frame(jpg.tobytes())

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

            if now - last_metrics >= 0.1:
                last_metrics = now
                hub.push(self._metrics(now, face, ear, mar, pitch, fps))

    def _detect(self, now, ear, mar, pitch):
        f = self.fired
        # eyes
        if ear < self.ear_thresh:
            self.closed_since = self.closed_since or now
            d = now - self.closed_since
            if d >= EYE_SIGN_SEC and not f["eye_sign"]:
                f["eye_sign"] = True
                self.on_sign(now, f"eyes closed {d:.1f} s")
            if d >= EYE_ALARM_SEC and now - self.last_alert >= 3.0:
                self.alert(
                    now, f"eyes closed {d:.1f} s"
                )  # repeats while eyes stay shut
        else:
            self.closed_since = None
            f["eye_sign"] = False
        # yawn
        if mar > MAR_YAWN:
            self.yawn_since = self.yawn_since or now
            if now - self.yawn_since >= YAWN_SEC and not f["yawn"]:
                f["yawn"] = True
                self.on_sign(now, "yawning")
        else:
            self.yawn_since, f["yawn"] = None, False
        # head nod
        if pitch > self.pitch_base + NOD_DELTA:
            self.nod_since = self.nod_since or now
            if now - self.nod_since >= NOD_SEC and not f["nod"]:
                f["nod"] = True
                self.on_sign(now, "head nodding")
        else:
            self.nod_since, f["nod"] = None, False

    def _draw_face(self, frame, pts, eyes_closed, now):
        ex = [pts[i][0] for i in LEFT_EYE + RIGHT_EYE]
        ey = [pts[i][1] for i in LEFT_EYE + RIGHT_EYE]
        x0, y0 = int(min(ex)) - 14, int(min(ey)) - 14
        x1, y1 = int(max(ex)) + 14, int(max(ey)) + 14
        color = (122, 18, 224) if eyes_closed else (120, 210, 120)  # BGR pink / green
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        if eyes_closed:
            txt = f"Eyes closed: {now - self.closed_since:.1f} s"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x0, y1 + 4), (x0 + tw + 12, y1 + th + 16), color, -1)
            cv2.putText(
                frame,
                txt,
                (x0 + 6, y1 + th + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    def _metrics(self, now, has_face, ear, mar, pitch, fps):
        if self.ear_thresh is None:
            stage = "calibrating"
        elif now < self.alert_until:
            stage = "alert"
        elif now < self.nudge_until:
            stage = "nudge"
        elif not has_face:
            stage = "noface"
        elif (
            (self.closed_since and now - self.closed_since > 0.4)
            or self.yawn_since
            or self.nod_since
        ):
            stage = "detect"
        else:
            stage = "watch"
        calibrated = self.ear_thresh is not None and has_face
        return {
            "type": "metrics",
            "stage": stage,
            "face": has_face,
            "calib_left": round(
                max(0.0, CALIBRATION_SEC - (now - self.calib_start)), 1
            ),
            "closed": round(now - self.closed_since, 1) if self.closed_since else 0,
            "eye": round(ear / self.open_ear, 3) if calibrated else None,
            "eye_thr": EAR_RATIO,
            "mouth": round(mar / MAR_YAWN, 3) if calibrated else None,
            "head": round((pitch - self.pitch_base) / NOD_DELTA, 3)
            if calibrated
            else None,
            "fps": round(fps, 1),
            "nudges": self.nudges,
            "alerts": self.alerts,
        }


# ---------------------------------------------------------------- web server
def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/stream":
                self._stream()
            elif path == "/events":
                self._events()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/recalibrate":
                hub.recalibrate.set()
            elif path == "/api/test-nudge":
                hub.log("Test: voice nudge", "nudge")
                hub.push({"type": "nudge", "text": NUDGE_TEXT, "reason": "test"})
            elif path == "/api/test-alert":
                hub.log("Test: alarm", "alert")
                hub.push({"type": "alert", "reason": "test"})
            else:
                return self._send(404, b"not found", "text/plain")
            self._send(200, b"ok", "text/plain")

        def _stream(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            fid = -1
            try:
                while True:
                    fid, jpg = hub.wait_frame(fid)
                    if jpg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last, backlog = hub.snapshot()
            try:
                for ev in backlog:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                while True:
                    evs, last = hub.wait_events(last, 1.0)
                    if not evs:
                        self.wfile.write(b": ping\n\n")
                    for ev in evs:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def local_addresses():
    addrs = []
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        ).stdout
        addrs = [a for a in out.split() if "." in a]
    except Exception:
        pass
    return addrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--port", type=int, default=8000, help="web port (default 8000)")
    ap.add_argument(
        "--pi-audio", action="store_true", help="also play sounds on the Pi"
    )
    args = ap.parse_args()

    hub = Hub()
    detector = Detector(hub, args.camera, PiAudio() if args.pi_audio else None)
    threading.Thread(target=detector.run, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(hub))
    server.daemon_threads = True
    print("\nDrowsiness Companion is running. Open on the laptop:")
    print(f"   http://{socket.gethostname()}.local:{args.port}")
    for a in local_addresses():
        print(f"   http://{a}:{args.port}")
    print("Ctrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- dashboard page
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drowsiness Companion</title>
<style>
:root{--navy:#22398a;--ink:#1b2340;--muted:#69718c;--bg:#f3f5fa;--card:#fff;--line:#e2e6f0;
--blue-soft:#e9effb;--orange:#d98b00;--orange-soft:#fff3dd;--pink:#e0127a;--pink-soft:#fde4f0;--green:#1f9d6b}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 24px;background:var(--card);border-bottom:1px solid var(--line)}
h1{margin:0;font-size:22px;color:var(--navy);letter-spacing:-.01em}
h1 small{font-weight:500;color:var(--muted);font-size:14px;margin-left:8px}
.conn{font-size:13px;color:var(--muted);white-space:nowrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#bbb;margin-right:6px;vertical-align:1px}
main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr);gap:20px;padding:20px 24px;max-width:1500px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr;padding:16px}}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.video{position:relative;background:#0f1630;border-radius:14px;overflow:hidden;aspect-ratio:4/3;transition:box-shadow .2s}
.video img{width:100%;height:100%;object-fit:contain;display:block}
.video.alerting{box-shadow:0 0 0 6px var(--pink)}
.bubble{position:absolute;left:50%;top:18px;transform:translate(-50%,-12px);opacity:0;transition:.25s;pointer-events:none;
background:#fff;color:var(--navy);padding:14px 24px;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.3);text-align:center;font-weight:700;font-size:22px;max-width:90%}
.bubble.show{opacity:1;transform:translate(-50%,0)}
.bubble small{display:block;font-weight:500;font-size:17px;margin-top:3px}
.bubble.alert{background:var(--pink);color:#fff;animation:flash .5s steps(2) infinite}
@keyframes flash{50%{background:#9e0b55}}
.status{font-size:15px;margin:0 0 10px;color:var(--muted);min-height:20px}
.status b{color:var(--ink)}
.stages{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.stage{border-radius:12px;padding:12px 6px 10px;text-align:center;border:2px solid var(--line);background:var(--card);opacity:.45;transition:.2s}
.stage .n{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;color:#fff;font-size:12px;font-weight:700;background:var(--navy)}
.stage b{display:block;font-size:15px;margin-top:6px;color:var(--navy)}
.stage span{display:block;font-size:11px;color:var(--muted);margin-top:2px;line-height:1.3}
.stage.on{opacity:1}
.stage.watch.on,.stage.detect.on{background:var(--blue-soft);border-color:var(--navy)}
.stage.nudge .n{background:var(--orange)} .stage.nudge b{color:var(--orange)}
.stage.nudge.on{background:var(--orange-soft);border-color:var(--orange)}
.stage.alert .n{background:var(--pink)} .stage.alert b{color:var(--pink)}
.stage.alert.on{background:var(--pink-soft);border-color:var(--pink)}
.big{display:flex;gap:12px}
.big div{flex:1;background:var(--bg);border-radius:10px;padding:10px 12px}
.big b{display:block;font-size:26px;font-variant-numeric:tabular-nums}
.big span{font-size:12px;color:var(--muted)}
.meter{margin-top:12px}
.meter label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.meter label em{font-style:normal;color:var(--muted);font-variant-numeric:tabular-nums}
.bar{position:relative;height:10px;background:var(--bg);border-radius:6px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--navy);border-radius:6px;transition:width .1s}
.bar i.hot{background:var(--pink)}
.bar u{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink);opacity:.5}
.log{list-style:none;margin:0;padding:0;max-height:260px;overflow:auto;font-size:13px}
.log li{display:flex;gap:10px;padding:6px 0;border-top:1px solid var(--line)}
.log li:first-child{border-top:0}
.log time{color:var(--muted);font-variant-numeric:tabular-nums}
.log li.nudge span{color:var(--orange);font-weight:600}
.log li.alert span{color:var(--pink);font-weight:700}
.log li.warn span{color:#b25b00}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;border-radius:10px;cursor:pointer}
button:hover{border-color:var(--navy)}
#start{position:fixed;inset:0;z-index:10;display:grid;place-items:center;background:rgba(20,28,60,.72);border:0;border-radius:0;color:#fff;cursor:pointer}
#start div{background:var(--card);color:var(--ink);padding:28px 36px;border-radius:18px;text-align:center;max-width:420px}
#start b{display:block;font-size:22px;color:var(--navy);margin-bottom:6px}
</style></head>
<body>
<button id="start"><div><b>Click to start</b>This lets the laptop speakers play the voice nudge and the alarm.</div></button>
<header>
  <h1>Drowsiness Companion<small>Raspberry Pi 5 &middot; live</small></h1>
  <div class="conn"><span class="dot" id="dot"></span><span id="conn">Connecting&hellip;</span></div>
</header>
<main>
  <div class="col">
    <div class="video" id="video">
      <img id="cam" src="/stream" alt="Live camera">
      <div class="bubble" id="bubble"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <p class="status" id="status">Waiting for the Pi&hellip;</p>
      <div class="stages">
        <div class="stage watch" id="s-watch"><span class="n">1</span><b>Watch</b><span>Tracking face and eyes</span></div>
        <div class="stage detect" id="s-detect"><span class="n">2</span><b>Detect</b><span>Long blink, yawn, nod</span></div>
        <div class="stage nudge" id="s-nudge"><span class="n">3</span><b>Nudge</b><span>&ldquo;Time for a break?&rdquo;</span></div>
        <div class="stage alert" id="s-alert"><span class="n">4</span><b>Alert</b><span>Loud alarm</span></div>
      </div>
    </div>
    <div class="card">
      <h2>Live signals</h2>
      <div class="big">
        <div><b id="closed">0.0 s</b><span>Eyes closed</span></div>
        <div><b id="nudges">0</b><span>Nudges</span></div>
        <div><b id="alerts">0</b><span>Alerts</span></div>
      </div>
      <div class="meter"><label>Eye openness <em id="eyeV">&ndash;</em></label><div class="bar"><i id="eyeB"></i><u id="eyeT"></u></div></div>
      <div class="meter"><label>Mouth opening (yawn) <em id="mouthV">&ndash;</em></label><div class="bar"><i id="mouthB"></i><u style="left:62.5%"></u></div></div>
      <div class="meter"><label>Head drop (nod) <em id="headV">&ndash;</em></label><div class="bar"><i id="headB"></i><u style="left:50%"></u></div></div>
    </div>
    <div class="card">
      <h2>Event log</h2>
      <ul class="log" id="log"></ul>
    </div>
    <div class="btns">
      <button id="recal">Recalibrate</button>
      <button id="tnudge">Test voice</button>
      <button id="talert">Test alarm</button>
      <span class="conn" id="fps" style="align-self:center;margin-left:auto"></span>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let actx = null, voice = null, bubbleTimer = null;

function pickVoice(){
  const vs = speechSynthesis.getVoices().filter(v => v.lang && v.lang.startsWith('en'));
  voice = vs.find(v => v.localService) || vs[0] || null;
}
if ('speechSynthesis' in window){ speechSynthesis.onvoiceschanged = pickVoice; pickVoice(); }

$('start').onclick = () => {
  try { actx = new (window.AudioContext || window.webkitAudioContext)(); actx.resume(); } catch(e){}
  try { speechSynthesis.speak(new SpeechSynthesisUtterance(' ')); } catch(e){}
  $('start').remove();
};

function speak(text){
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95; u.volume = 1; if (voice) u.voice = voice;
    speechSynthesis.speak(u);
  } catch(e){}
}
function siren(sec){
  if (!actx) return;
  try { speechSynthesis.cancel(); } catch(e){}
  const o = actx.createOscillator(), g = actx.createGain(), t0 = actx.currentTime;
  o.type = 'square'; g.gain.value = 0.25;
  o.connect(g).connect(actx.destination);
  for (let i = 0; i < sec * 4; i++) o.frequency.setValueAtTime(i % 2 ? 1320 : 880, t0 + i * 0.25);
  o.start(t0); o.stop(t0 + sec);
}
function bubble(kind, main, sub){
  const b = $('bubble');
  b.className = 'bubble show' + (kind === 'alert' ? ' alert' : '');
  b.innerHTML = main + (sub ? '<small>' + sub + '</small>' : '');
  $('video').classList.toggle('alerting', kind === 'alert');
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => { b.className = 'bubble'; $('video').classList.remove('alerting'); }, 4000);
}
function addLog(ev){
  const li = document.createElement('li');
  li.className = ev.level;
  const t = document.createElement('time'); t.textContent = ev.t;
  const s = document.createElement('span'); s.textContent = ev.msg;
  li.append(t, s);
  const log = $('log'); log.prepend(li);
  while (log.children.length > 60) log.lastChild.remove();
}
function meter(id, v, max, hotAt){
  const bar = $(id + 'B'), txt = $(id + 'V');
  if (v === null || v === undefined){ bar.style.width = '0'; txt.textContent = '–'; return; }
  bar.style.width = Math.max(0, Math.min(1, v / max)) * 100 + '%';
  bar.classList.toggle('hot', hotAt(v));
  txt.textContent = Math.round(v * 100) + '%';
}
const STATUS = {
  calibrating: m => m.face ? `<b>Calibrating</b>: keep your eyes open and look ahead (${m.calib_left.toFixed(1)} s)` : '<b>Calibrating</b>: no face in view yet',
  noface: () => '<b>No face in view</b>: sit in front of the camera',
  watch: () => '<b>Watching</b>: driver looks alert',
  detect: () => '<b>Detecting</b>: possible drowsiness sign',
  nudge: () => '<b>Nudge</b>: gentle reminder given',
  alert: () => '<b>ALERT</b>: driver is drowsy'
};
const ORDER = ['watch', 'detect', 'nudge', 'alert'];
function onMetrics(m){
  $('status').innerHTML = (STATUS[m.stage] || (() => ''))(m);
  const idx = ORDER.indexOf(m.stage);
  ORDER.forEach((s, i) => $('s-' + s).classList.toggle('on', idx >= 0 && i === idx));
  $('closed').textContent = m.closed.toFixed(1) + ' s';
  $('closed').style.color = m.closed > 0 ? 'var(--pink)' : '';
  $('nudges').textContent = m.nudges; $('alerts').textContent = m.alerts;
  $('eyeT').style.left = (m.eye_thr / 1.3 * 100) + '%';
  meter('eye', m.eye, 1.3, v => v < m.eye_thr);
  meter('mouth', m.mouth, 1.6, v => v > 1);
  meter('head', m.head === null ? null : Math.max(0, m.head), 2, v => v > 1);
  $('fps').textContent = m.fps + ' fps';
}

const es = new EventSource('/events');
es.onopen = () => { $('dot').style.background = 'var(--green)'; $('conn').textContent = 'Connected to Pi'; };
es.onerror = () => { $('dot').style.background = 'var(--pink)'; $('conn').textContent = 'Reconnecting to Pi…'; };
es.onmessage = e => {
  const ev = JSON.parse(e.data);
  if (ev.type === 'metrics') onMetrics(ev);
  else if (ev.type === 'log') addLog(ev);
  else if (ev.type === 'nudge'){ speak(ev.text); bubble('nudge', 'Hey, you look tired.', 'Time for a break?'); }
  else if (ev.type === 'alert'){ siren(2.5); bubble('alert', 'WAKE UP!', 'Pull over and rest'); }
};

const cam = $('cam');
cam.onerror = () => setTimeout(() => { cam.src = '/stream?' + Date.now(); }, 1000);

const post = p => fetch(p, {method: 'POST'}).catch(() => {});
$('recal').onclick = () => post('/api/recalibrate');
$('tnudge').onclick = () => post('/api/test-nudge');
$('talert').onclick = () => post('/api/test-alert');
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
