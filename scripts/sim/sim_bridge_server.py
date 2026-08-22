"""Runs a SimplerEnv (ManiSkill2_real2sim / SAPIEN 2.2.2) rollout as a local socket server.

This process is meant to be launched with the *Python 3.11* interpreter in
`.venv-sim` (SAPIEN 2.2.2 has no wheels for Python 3.12, and its bindings
predate NumPy 2.0's C-API change, so it needs its own isolated environment
-- see scripts/sim/setup_sim_env.sh). It speaks a tiny length-prefixed JSON
protocol over a local TCP socket so a VLA model running in a *different*
Python environment (e.g. this repo's main .venv) can drive it in a closed
control loop: request a frame, predict an action, send the action back,
get the next frame.

Wire format: each message (both directions) is `<4-byte big-endian length><JSON bytes>`.
Commands (client -> server), one JSON object per message:
    {"cmd": "reset"}
    {"cmd": "step", "action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}
    {"cmd": "close"}
Responses (server -> client):
    {"ok": true, "image_b64": "<PNG>", "instruction": "...", "reward": 0.0,
     "terminated": false, "truncated": false, "success": false, "info": {...}}
    {"ok": false, "error": "..."}

On startup the server prints a line "READY <port>" to stdout once it has
built the environment and is listening -- everything else SAPIEN/ManiSkill2
print to stdout/stderr during setup is noise the client should ignore; only
that one line matters for the handshake. All protocol traffic after that
goes over the socket, never stdio, so later library log spam can't corrupt it.
"""

import argparse
import base64
import io
import json
import os
import socket
import struct
import sys
import traceback

# Headless Vulkan needs the NVIDIA ICD explicitly pointed to -- the loader
# doesn't always find it under /etc/vulkan/icd.d on its own in this setup.
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", "")


def log(msg):
    print(f"[sim_bridge_server] {msg}", file=sys.stderr, flush=True)


def send_msg(conn, obj):
    payload = json.dumps(obj, default=_json_default).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    header = recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _json_default(o):
    # Best-effort conversion for the numpy scalars/arrays that show up in
    # ManiSkill2's info dicts -- never let a serialization hiccup kill the
    # whole rollout, worst case a field becomes a string.
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def encode_image(rgb_array):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_obs_response(env, obs, reward=0.0, terminated=False, truncated=False, success=False, info=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    image = get_image_from_maniskill2_obs_dict(env, obs)
    return {
        "ok": True,
        "image_b64": encode_image(image),
        "instruction": env.get_language_instruction(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "info": info or {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="simpler_env task name, e.g. widowx_put_eggplant_in_basket")
    parser.add_argument("--port", type=int, default=0, help="0 = pick any free port")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    log(f"building task '{args.task}' ...")
    import simpler_env

    if args.task not in simpler_env.ENVIRONMENTS:
        log(f"FATAL: unknown task '{args.task}'. Available: {simpler_env.ENVIRONMENTS}")
        sys.exit(1)

    env = simpler_env.make(args.task)
    log("env built")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)
    bound_port = server_sock.getsockname()[1]

    # The one line the client actually parses -- flush immediately.
    print(f"READY {bound_port}", flush=True)
    log(f"listening on {args.host}:{bound_port}")

    conn, addr = server_sock.accept()
    log(f"client connected from {addr}")

    try:
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    log("client disconnected")
                    break
                cmd = msg.get("cmd")
                try:
                    if cmd == "reset":
                        obs, reset_info = env.reset()
                        resp = build_obs_response(env, obs, info={"reset_info": _shallow_str(reset_info)})
                    elif cmd == "step":
                        import numpy as np

                        action = np.array(msg["action"], dtype=np.float32)
                        obs, reward, terminated, truncated, info = env.step(action)
                        success = bool(info.get("success", terminated))
                        resp = build_obs_response(
                            env, obs, reward=reward, terminated=terminated, truncated=truncated,
                            success=success, info={"raw_terminated": bool(terminated)},
                        )
                    elif cmd == "close":
                        send_msg(conn, {"ok": True})
                        break
                    else:
                        resp = {"ok": False, "error": f"unknown cmd {cmd!r}"}
                    if cmd != "close":
                        send_msg(conn, resp)
                except Exception as err:
                    log("error handling command:\n" + traceback.format_exc())
                    try:
                        send_msg(conn, {"ok": False, "error": f"{type(err).__name__}: {err}"})
                    except Exception:
                        break
    finally:
        server_sock.close()
        log("server shut down")


def _shallow_str(obj):
    # reset_info can contain non-JSON-friendly objects; keep only a printable summary.
    try:
        json.dumps(obj, default=_json_default)
        return obj
    except Exception:
        return str(obj)


if __name__ == "__main__":
    main()
