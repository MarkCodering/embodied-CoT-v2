"""Client for scripts/sim/sim_bridge_server.py.

Launches the SimplerEnv physics sim as a subprocess under the *separate*
Python 3.11 `.venv-sim` environment (see setup_sim_env.sh for why) and talks
to it over a local socket, so a VLA model loaded in this repo's main .venv
(Python 3.12) can drive a real closed-loop rollout without either
environment's dependencies (transformers vs. sapien==2.2.2 + numpy<2)
colliding.

Typical use from the notebook:

    from scripts.sim.sim_bridge_client import SimplerEnvBridge

    with SimplerEnvBridge(task="widowx_put_eggplant_in_basket") as sim:
        image, instruction = sim.reset()
        for _ in range(60):
            inputs = processor(get_openvla_prompt(instruction), image).to(device, dtype=model_dtype)
            action, _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            image, reward, terminated, truncated, success, info = sim.step(action)
            if terminated or truncated:
                break
"""

import base64
import io
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VENV_PYTHON = REPO_ROOT / ".venv-sim" / "bin" / "python"
SERVER_SCRIPT = Path(__file__).resolve().with_name("sim_bridge_server.py")


class SimplerEnvBridgeError(RuntimeError):
    pass


class SimplerEnvBridge:
    def __init__(self, task, startup_timeout=180.0, sim_python=None):
        self.task = task
        self.startup_timeout = startup_timeout
        self.sim_python = Path(sim_python) if sim_python else SIM_VENV_PYTHON
        self._proc = None
        self._sock = None

        if not self.sim_python.exists():
            raise SimplerEnvBridgeError(
                f"Sim Python not found at {self.sim_python}. Run "
                f"`bash scripts/sim/setup_sim_env.sh` once to provision the SimplerEnv environment."
            )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -- lifecycle -----------------------------------------------------
    def start(self):
        self._proc = subprocess.Popen(
            [str(self.sim_python), str(SERVER_SCRIPT), "--task", self.task, "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        port = self._wait_for_ready()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(("127.0.0.1", port))
        return self

    def _wait_for_ready(self):
        deadline = time.time() + self.startup_timeout
        stderr_tail = []
        while time.time() < deadline:
            if self._proc.poll() is not None:
                stderr_tail = self._proc.stderr.read()
                raise SimplerEnvBridgeError(
                    f"sim server exited early (code {self._proc.returncode}) before signaling ready.\n"
                    f"stderr:\n{stderr_tail}"
                )
            line = self._proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("READY "):
                return int(line.split()[1])
            # Anything else is SAPIEN/ManiSkill2 startup noise -- surface it
            # for debugging but don't treat it as fatal.
            print(f"[sim setup] {line}", file=sys.stderr)
        self._proc.kill()
        raise SimplerEnvBridgeError(f"sim server did not become ready within {self.startup_timeout}s")

    def close(self):
        if self._sock is not None:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    # -- protocol --------------------------------------------------------
    def _send(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(payload)) + payload)

    def _recv(self):
        header = self._recv_exact(4)
        (length,) = struct.unpack(">I", header)
        payload = self._recv_exact(length)
        return json.loads(payload.decode("utf-8"))

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise SimplerEnvBridgeError("sim server connection closed unexpectedly")
            buf += chunk
        return buf

    def _roundtrip(self, obj):
        self._send(obj)
        resp = self._recv()
        if not resp.get("ok", False):
            raise SimplerEnvBridgeError(f"sim server error: {resp.get('error')}")
        return resp

    @staticmethod
    def _decode_image(image_b64):
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

    # -- rollout API -------------------------------------------------------
    def reset(self):
        resp = self._roundtrip({"cmd": "reset"})
        return self._decode_image(resp["image_b64"]), resp["instruction"]

    def step(self, action):
        action = list(float(a) for a in action)
        resp = self._roundtrip({"cmd": "step", "action": action})
        image = self._decode_image(resp["image_b64"])
        return (
            image,
            resp["reward"],
            resp["terminated"],
            resp["truncated"],
            resp["success"],
            resp["info"],
        )
