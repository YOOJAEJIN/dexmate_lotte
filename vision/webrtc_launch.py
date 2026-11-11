import argparse
import asyncio
import fractions
import json
import os
import signal
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
import av


sys.path.append("/usr/lib/python3/dist-packages")

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst

Gst.init(None)


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>UDP to WebRTC Stream</title>
    <style>
    body {
        font-family: sans-serif;
        margin: 0;
        padding: 0;
        background: #101418;
        color: #f5f7fa;
        text-align: center;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .media {
        width: min(96vw, 1600px);
        height: min(96vh, 900px);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    video {
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: #000;
        border-radius: 8px;
    }
    </style>
</head>
<body>
    <div class="media">
        <video id="video" autoplay playsinline muted></video>
    </div>
    <script src="client.js"></script>
</body>
</html>
"""


CLIENT_JS = """
let pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    return pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .then(() => new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        }))
        .then(() => fetch('offer', {  // relative: works at / and behind /cam/<port>/ proxy
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pc.localDescription)
        }))
        .then((response) => response.json())
        .then((answer) => pc.setRemoteDescription(answer))
        .catch((e) => alert(e));
}

function start() {
    pc = new RTCPeerConnection({ sdpSemantics: 'unified-plan' });
    pc.addEventListener('track', (evt) => {
        document.getElementById('video').srcObject = evt.streams[0];
    });
    negotiate();
}

function stop() {
    if (pc) {
        pc.close();
        pc = null;
    }
}

window.onload = start;
window.onbeforeunload = stop;
"""


class GstReceiver:
    def __init__(self, udp_port: int):
        pipeline_str = f"""
        udpsrc port={udp_port} caps="application/x-rtp,media=video,encoding-name=H264,payload=96" !
        rtph264depay !
        h264parse !
        avdec_h264 !
        videoconvert !
        video/x-raw,format=BGR !
        appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def get_frame(self) -> Optional[np.ndarray]:
        sample = self.appsink.emit("pull-sample")
        if sample is None:
            return None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()
        buffer.unmap(map_info)
        return frame

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


class BGRArrayVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._start_time: Optional[float] = None

    async def recv(self) -> av.VideoFrame:
        return await self._queue.get()

    def push_frame(self, bgr_frame: np.ndarray, loop: asyncio.AbstractEventLoop):
        try:
            vf = av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
            if self._start_time is None:
                self._start_time = time.time()
                pts = 0
            else:
                pts = int((time.time() - self._start_time) * 90000)
            vf.pts = pts
            vf.time_base = fractions.Fraction(1, 90000)
        except Exception:
            return

        def _put():
            try:
                if self._queue.full():
                    self._queue.get_nowait()
                self._queue.put_nowait(vf)
            except Exception:
                pass

        if not loop.is_closed():
            loop.call_soon_threadsafe(_put)


class UdpToWebRtcServer:
    def __init__(self, udp_port: int, host: str, http_port: int, cert_path: Path, key_path: Path):
        self._udp_port = udp_port
        self._host = host
        self._http_port = http_port
        self._cert_path = cert_path
        self._key_path = key_path

        self._receiver = GstReceiver(self._udp_port)
        self._track = BGRArrayVideoStreamTrack()
        self._relay = MediaRelay()

        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._app = web.Application()
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/client.js", self._client_js)
        self._app.router.add_post("/offer", self._offer)
        self._app.router.add_options("/offer", self._options)

        self._pcs = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(content_type="text/html", text=INDEX_HTML)

    async def _client_js(self, request: web.Request) -> web.Response:
        return web.Response(content_type="application/javascript", text=CLIENT_JS)

    async def _options(self, request: web.Request) -> web.Response:
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    async def _offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)
        relayed_track = self._relay.subscribe(self._track)
        pc.addTransceiver(relayed_track, direction="sendonly")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await self._cleanup_pc(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                }
            ),
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    def _capture_loop(self):
        # Optional frame tap: SEG_TAP_PORT=<port> publishes decoded BGR frames
        # over zmq (CONFLATE: subscribers always get the latest frame only).
        tap = None
        tap_port = os.getenv("SEG_TAP_PORT")
        if tap_port:
            import zmq
            ctx = zmq.Context.instance()
            tap = ctx.socket(zmq.PUB)
            tap.setsockopt(zmq.CONFLATE, 1)
            tap.bind(f"tcp://127.0.0.1:{tap_port}")
            print(f"TAP   : zmq pub tcp://127.0.0.1:{tap_port}")

        while not self._stop_event.is_set():
            frame = self._receiver.get_frame()
            if frame is not None and self._loop is not None:
                self._track.push_frame(frame, self._loop)
                if tap is not None:
                    try:
                        h, w = frame.shape[:2]
                        tap.send(w.to_bytes(2, "big") + h.to_bytes(2, "big") + frame.tobytes(),
                                 zmq.NOBLOCK)
                    except Exception:
                        pass
            else:
                time.sleep(0.002)

    async def _cleanup_pc(self, pc: RTCPeerConnection):
        self._pcs.discard(pc)
        try:
            await pc.close()
        except Exception:
            pass

    async def _shutdown(self):
        for pc in list(self._pcs):
            await self._cleanup_pc(pc)

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        runner = web.AppRunner(self._app)
        await runner.setup()

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(str(self._cert_path), str(self._key_path))
        site = web.TCPSite(runner, self._host, self._http_port, ssl_context=ssl_context)
        await site.start()

        print(f"HTTPS : {self._http_port}")
        print(f"UDP   : {self._udp_port}")
        print(f"URL   : https://{self._host if self._host != '0.0.0.0' else '127.0.0.1'}:{self._http_port}")

        while not self._stop_event.is_set():
            await asyncio.sleep(0.2)

        await self._shutdown()
        await runner.cleanup()
        self._receiver.stop()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

    def stop(self):
        self._stop_event.set()


def resolve_cert_paths(cert_arg: Optional[str], key_arg: Optional[str]) -> tuple[Path, Path]:
    env_cert = os.getenv("XR_TELEOP_CERT")
    env_key = os.getenv("XR_TELEOP_KEY")

    default_cert = Path("/home/dexmate/certs/cert.pem")
    default_key = Path("/home/dexmate/certs/key.pem")

    cert_path = Path(cert_arg or env_cert or default_cert).expanduser().resolve()
    key_path = Path(key_arg or env_key or default_key).expanduser().resolve()
    if not cert_path.exists() or not key_path.exists():
        raise FileNotFoundError(
            f"Certificate not found. cert={cert_path}, key={key_path}. "
            "Use --cert/--key or XR_TELEOP_CERT/XR_TELEOP_KEY."
        )
    return cert_path, key_path


def main():
    parser = argparse.ArgumentParser(description="UDP H264 to WebRTC HTTPS relay")
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--udp-port", type=int, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--cert", type=str, default=None)
    parser.add_argument("--key", type=str, default=None)
    args = parser.parse_args()

    cert_path, key_path = resolve_cert_paths(args.cert, args.key)
    server = UdpToWebRtcServer(
        udp_port=args.udp_port,
        host=args.host,
        http_port=args.http_port,
        cert_path=cert_path,
        key_path=key_path,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop_handler(signum, _frame):
        print(f"Received signal {signum}, stopping...")
        server.stop()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    try:
        loop.run_until_complete(server.run())
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()