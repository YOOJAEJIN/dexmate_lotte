#!/usr/bin/env bash
# Bring up the HEAD SEG camera pipeline. Idempotent: parts already up are skipped,
# so it is safe to run by hand at any time and from cron at boot:
#   @reboot <repo>/vision/start_head_seg.sh >> <repo>/vision/logs/autostart.log 2>&1
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANO="${NANO:-dexmate-nano@192.168.50.22}"
NANO_PY="${NANO_PY:-/home/dexmate-nano/miniconda3/bin/python}"
NANO_SENDER="${NANO_SENDER:-gstsender/gstsender.py}"  # relative to the nano's $HOME
RELAY_PY="${RELAY_PY:-/home/dexmate/webrtc/webrtc/bin/python}"
SEG_PY="${SEG_PY:-/home/dexmate/webrtc/segenv/bin/python}"
# seg_overlay reads the head IMU (gravity) and robot/joints over zenoh. Without
# ZENOH_CONFIG dexcomm falls back to mDNS discovery and may never connect.
# Default: the robot's own config ~/.dexmate/comm/zenoh/<robot>.dzcfg, whose
# name is the robot name with "/" -> "_" (dm_xxx-1p.dzcfg -> dm/xxx-1p).
if [ -z "${ZENOH_CONFIG:-}" ]; then
  ZENOH_CONFIG="$(ls "$HOME"/.dexmate/comm/zenoh/*.dzcfg 2>/dev/null | head -1)"
fi
if [ -z "${ROBOT_NAME:-}" ] && [ -n "$ZENOH_CONFIG" ]; then
  ROBOT_NAME="$(basename "$ZENOH_CONFIG" .dzcfg | sed 's|_|/|')"
fi
export ROBOT_NAME ZENOH_CONFIG
LOG="$HERE/logs"
mkdir -p "$LOG"

log() { echo "[$(date '+%F %T')] $*"; }
listening() { ss -tln | grep -q ":$1 "; }
nano() { timeout "${2:-10}" ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$NANO" "$1"; }
nano_up() { nano "ss -tln | grep -q ':55703 '"; }  # metric-depth port = patched sender is up

# 1. nano ZED sender. At boot the nano may still be coming up, so wait up to 3 min.
for i in $(seq 1 36); do nano true && break; log "waiting for nano ($i/36)"; sleep 5; done
if nano_up; then
  log "nano sender already up"
else
  for attempt in 1 2 3; do
    log "starting nano sender (attempt $attempt/3)"
    # ssh keeps the channel open after the remote job is backgrounded; the timeout
    # only ends ssh, the sender keeps running.
    nano "cd ~ && setsid nohup $NANO_PY $NANO_SENDER > /tmp/gstsender.log 2>&1 < /dev/null &" 8 || true
    for _ in $(seq 1 20); do nano_up && break 2; sleep 2; done
    log "sender not up yet (ZED busy or still booting): $(nano 'tail -1 /tmp/gstsender.log' || true)"
    sleep 10
  done
  nano_up && log "nano sender up" || log "WARN: nano sender failed; continuing (relays/seg wait for frames)"
fi

# 2. Thor WebRTC relays 60001..60006. Only 60001 publishes the frame tap: giving
#    every relay the same SEG_TAP_PORT makes all but one fail to bind.
cd "$HERE"
for n in 1 2 3 4 5 6; do
  port="6000$n"
  # A 60001 relay without the tap (e.g. started by ~/webrtc/run_webrtc_from_robot_config.sh,
  # 2026-11-16) streams fine but HEAD SEG never gets a frame - replace it with ours.
  if [ "$n" -eq 1 ] && listening "$port" && ! listening 55601; then
    pid="$(ss -tlnpH "sport = :$port" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
    log "relay $port is up without the HEAD SEG tap (pid ${pid:-?}) - restarting it with the tap"
    [ -n "$pid" ] && kill "$pid"
    for _ in $(seq 1 25); do listening "$port" || break; sleep 0.2; done
  fi
  if listening "$port"; then log "relay $port already up"; continue; fi
  tap=""; [ "$n" -eq 1 ] && tap="SEG_TAP_PORT=55601"
  log "starting relay $port"
  env $tap setsid nohup "$RELAY_PY" webrtc_launch.py --http-port "$port" --udp-port "5000$n" \
    > "$LOG/webrtc_$port.log" 2>&1 < /dev/null &
done

# 3. Segmentation overlay. It blocks on its own until frames arrive, so no wait here.
if pgrep -f "[s]eg_overlay\.py" > /dev/null; then
  log "seg_overlay already up"
else
  log "starting seg_overlay"
  setsid nohup "$SEG_PY" seg_overlay.py > "$LOG/seg_overlay.log" 2>&1 < /dev/null &
fi
log "done"
