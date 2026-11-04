import asyncio
import cv2
import numpy as np
import threading
import time
import requests
from loguru import logger
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

class WebRTCReceiver:
    def __init__(self, url, name=None):
        self.url = url
        self.name = name or url
        self.pc = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.loop = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect())
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"WebRTC Receiver ({self.name}) error: {e}")
        finally:
            self.loop.run_until_complete(self._close())
            self.loop.close()

    async def _connect(self):
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                logger.info(f"WebRTC video track received from {self.name}")
                asyncio.ensure_future(self._handle_track(track))

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state for {self.name} is {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                await self.pc.close()
                self.running = False

        # 0. Add transceiver to receive video
        self.pc.addTransceiver("video", direction="recvonly")

        # 1. Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering to complete (up to 2 seconds)
        # This ensures the SDP contains candidates, helping the signaling server
        wait_start = time.time()
        while self.running and self.pc.iceGatheringState != "complete" and time.time() - wait_start < 2.0:
            await asyncio.sleep(0.1)

        if not self.running:
            return

        sdp = self.pc.localDescription.sdp
        sdp_type = self.pc.localDescription.type

        # 2. Send offer to signaling server and get answer
        try:
            # Note: We use verify=False because the user specified https://127.0.0.1
            # We use asyncio.to_thread to avoid blocking the event loop
            def send_offer():
                return requests.post(
                    self.url,
                    json={
                        "sdp": sdp,
                        "type": sdp_type
                    },
                    verify=False,
                    timeout=15.0  # Increased timeout for real robot network
                )

            response = await asyncio.to_thread(send_offer)
            response.raise_for_status()
            answer_data = response.json()
            answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
            await self.pc.setRemoteDescription(answer)
            logger.info(f"WebRTC connection established with {self.name} ({self.url})")
        except Exception as e:
            logger.error(f"Failed to connect to WebRTC signaling server ({self.name}): {e}")
            self.running = False

    async def _handle_track(self, track):
        while self.running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                with self.frame_lock:
                    self.latest_frame = img
                if self.latest_frame is not None and not hasattr(self, '_first_frame_logged'):
                    logger.info(f"First WebRTC frame received from {self.name}: {img.shape}")
                    self._first_frame_logged = True
            except Exception as e:
                logger.error(f"Error receiving WebRTC frame from {self.name}: {e}")
                break

    async def _close(self):
        if self.pc:
            await self.pc.close()
            self.pc = None

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame
