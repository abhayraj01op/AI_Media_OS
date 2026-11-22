"""
AI MEDIA OS — Installer

ComfyUI + SDXL (required by the face-lock tools) + Z-Image Turbo +
ReActor (face swap) + InstantID/PuLID/IPAdapter (face lock) + Real-ESRGAN
(upscale) + optional video generation.

Bug fixes applied after a real Colab run surfaced them:
  - ReActor's inswapper_128.onnx used to come from a repo that now
    requires authentication (401 Unauthorized) — switched to a public
    mirror.
  - PuLID's SDXL model filename was wrong (404) — the correct file in
    guozinan/PuLID is pulid_v1.1.safetensors, not
    ip-adapter_pulid_sdxl_fp16.safetensors (that name belongs to a
    DIFFERENT repo entirely).
  - core/downloader.py now flattens any filename that includes a subfolder
    (e.g. "split_files/vae/ae.safetensors") down to just its basename in
    the destination folder — ComfyUI's model browser only looks directly
    inside models/vae/, models/checkpoints/, etc., so a file that landed
    nested one level deeper than that (which is what happened before)
    would download "successfully" and still never show up as usable.
  - step_comfyui_core now checks for CUDA BEFORE downloading tens of GB of
    models, and gives one unmissable message instead of only discovering
    the problem after a 30-60 minute install: ComfyUI crashes on ANY
    non-GPU runtime with "AssertionError: Torch not compiled with CUDA
    enabled" — this happens when Colab's Runtime type isn't set to a GPU
    (Runtime > Change runtime type > T4 GPU), not from anything this
    installer does wrong.

Same proven patterns otherwise: real verification, cloudflared on
/content, ComfyUI-Manager network fetches disabled, idempotent start,
generous recheck-before-tunnel for ComfyUI.
"""

import os, sys, subprocess, time, json, threading, urllib.request, re, queue, shutil
from pathlib import Path

from core.state import ModuleState
from core.downloader import SmartDownloader


def _flag(name, default="0"):
    return os.environ.get(name, default) == "1"


class MediaInstaller:
    def __init__(self):
        self.base = Path("/content/drive/MyDrive/AI_Media_OS")
        self.logs = []
        self.state = None

    def log(self, msg):
        self.logs.append(msg)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @property
    def comfy(self):
        return self.base / "ComfyUI"

    def _save_state(self, cid, status, checks=None, detail=""):
        if not self.state:
            self.log(f"⚠️ Could not record {cid} status — actual result: {status.upper()} — {detail}")
            return
        self.state.save(cid, status, checks, detail)

    def _wait_for_http(self, url, timeout=90, interval=2):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=3)
                return True
            except Exception:
                time.sleep(interval)
        return False

    def _already_running(self, health_url, name):
        try:
            urllib.request.urlopen(health_url, timeout=3)
            self.log(f"{name} already running and responding — reusing it instead of starting a second copy.")
            return True
        except Exception:
            return False

    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            # torch might not be importable yet this early — fall back to
            # nvidia-smi, which is present on any GPU-enabled Colab runtime.
            return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0

    def _start_tunnel(self, cf_path, local_port, timeout=20):
        proc = subprocess.Popen([str(cf_path), "tunnel", "--url", f"http://localhost:{local_port}"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        q = queue.Queue()

        def _reader():
            try:
                for line in proc.stderr:
                    q.put(line)
            except Exception:
                pass

        threading.Thread(target=_reader, daemon=True).start()
        url = None
        deadline = time.time() + timeout
        buffer = ""
        while time.time() < deadline and url is None:
            try:
                line = q.get(timeout=0.5)
                buffer += line
                m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', buffer)
                if m:
                    url = m.group(0)
            except queue.Empty:
                continue
        return url, proc

    # ------------------------------------------------------------------
    def run(self):
        print("="*70)
        print("🚀 AI MEDIA OS — INSTALL")
        print("="*70)

        if not self._has_gpu():
            print("\n" + "🛑"*20)
            print("NO GPU DETECTED. This will download tens of GB of models and then")
            print("ComfyUI will still fail to start (torch has no CUDA to use).")
            print("FIX: Runtime menu (top) > Change runtime type > T4 GPU > Save,")
            print("     then Runtime > Run All again. This is a Colab setting, not")
            print("     something this installer can fix on its own.")
            print("🛑"*20 + "\n")
            print("Continuing anyway in 10 seconds in case this is a deliberate CPU-only test-run...")
            time.sleep(10)

        print("Default: SDXL + face-swap + face-lock + upscale (~12GB).")
        print("Optional (set env vars before this cell, see README): Z-Image Turbo,")
        print("FLUX.1-schnell, video generation.\n")

        steps = [
            ("Drive & Caches", self.step_drive),
            ("ComfyUI Core", self.step_comfyui_core),
            ("Face Tools (swap + lock)", self.step_face_tools),
            ("Upscaler", self.step_upscaler),
            ("Extra Image Models", self.step_extra_models),
            ("Video Generation (optional)", self.step_video),
            ("Start ComfyUI", self.step_start),
            ("Public Link", self.step_tunnel),
            ("Finalize", self.step_final),
        ]
        component_id = {
            "ComfyUI Core": "comfyui_core", "Face Tools (swap + lock)": "face_tools",
            "Upscaler": "upscaler", "Extra Image Models": "extra_models",
            "Video Generation (optional)": "video", "Start ComfyUI": "comfyui_running",
            "Public Link": "tunnel",
        }

        for name, func in steps:
            self.log(f"\n--- {name} ---")
            try:
                func()
                cid = component_id.get(name)
                if cid and self.state:
                    st = self.state.load(cid)
                    icon = {"ready": "✅", "partial": "🟡", "failed": "🔴", "skipped": "⏭️"}.get(st.get("status"), "❔")
                    self.log(f"{icon} {name}: {st.get('status', 'unknown').upper()} — {st.get('detail', '')}")
                else:
                    self.log(f"✅ {name}")
            except Exception as e:
                self.log(f"❌ {name}: {str(e)[:150]}")
                cid = component_id.get(name)
                if cid and self.state:
                    self._save_state(cid, "failed", detail=f"step raised: {str(e)[:150]}")

        self.show_done()

    # ------------------------------------------------------------------
    def step_drive(self):
        try:
            from google.colab import drive
            if not os.path.ismount("/content/drive"):
                drive.mount("/content/drive")
                time.sleep(3)
            else:
                self.log("Drive already mounted — skipping re-mount.")
        except ImportError:
            fallback = os.environ.get("AI_MEDIA_OS_HOME") or str(Path.home() / "AI_Media_OS")
            self.base = Path(fallback)
            self.log(f"Not running in Google Colab — using local path instead: {self.base}")
        except Exception as e:
            self.log(f"⚠️ Drive mount raised an error but continuing ({str(e)[:150]}).")

        folders = ["models/checkpoints", "models/vae", "models/text_encoders",
                   "models/diffusion_models", "models/loras", "models/controlnet",
                   "models/upscale_models", "models/insightface/models",
                   "models/instantid", "models/pulid",
                   "outputs", "logs", "state", "cache/pip", "cache/huggingface"]
        for f in folders:
            (self.base / f).mkdir(parents=True, exist_ok=True)
        self.log(f"Created folders under {self.base}")
        self.state = ModuleState(self.base / "state")

        os.environ["PIP_CACHE_DIR"] = str(self.base / "cache" / "pip")
        os.environ["HF_HOME"] = str(self.base / "cache" / "huggingface")

    # ------------------------------------------------------------------
    def step_comfyui_core(self):
        repo_already_present = self.comfy.exists()
        if not repo_already_present:
            self.comfy.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI", str(self.comfy)], check=True)
        os.chdir(str(self.comfy))
        pip_result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)

        mgr = self.comfy / "custom_nodes" / "ComfyUI-Manager"
        mgr.parent.mkdir(parents=True, exist_ok=True)
        if not mgr.exists():
            subprocess.run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI-Manager", str(mgr)], check=True)
        try:
            mgr_cfg_dir = self.comfy / "user" / "__manager"
            mgr_cfg_dir.mkdir(parents=True, exist_ok=True)
            mgr_cfg_path = mgr_cfg_dir / "config.ini"
            if not mgr_cfg_path.exists():
                mgr_cfg_path.write_text("[default]\nnetwork_mode = offline\n")
        except Exception:
            pass

        dl = SmartDownloader()
        sdxl_ok = dl.download(
            "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
            self.comfy / "models" / "checkpoints", min_size_mb=1000)
        if not sdxl_ok:
            self.log("⚠️ SDXL checkpoint download failed — image generation and ALL face tools "
                      "(they depend on SDXL) will not work until this is retried.")

        checks = {"repo_cloned": self.comfy.exists(), "requirements_installed": pip_result.returncode == 0, "sdxl": sdxl_ok}
        if all(checks.values()):
            status, detail = "ready", "ComfyUI cloned, deps installed, SDXL checkpoint verified."
        elif checks["repo_cloned"]:
            status, detail = "partial", "ComfyUI is present but SDXL and/or deps failed — see log above."
        else:
            status, detail = "failed", "ComfyUI repo failed to clone."
        self._save_state("comfyui_core", status, checks, detail)

    # ------------------------------------------------------------------
    def step_face_tools(self):
        custom_nodes = self.comfy / "custom_nodes"
        nodes = {
            "ComfyUI-ReActor": "https://github.com/Gourieff/ComfyUI-ReActor",
            "ComfyUI_InstantID": "https://github.com/cubiq/ComfyUI_InstantID",
            "PuLID_ComfyUI": "https://github.com/cubiq/PuLID_ComfyUI",
            "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        }
        clone_ok = {}
        for name, url in nodes.items():
            dest = custom_nodes / name
            if dest.exists():
                clone_ok[name] = True
                continue
            r = subprocess.run(["git", "clone", url, str(dest)], check=False)
            clone_ok[name] = r.returncode == 0
            req = dest / "requirements.txt"
            if req.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "insightface", "onnxruntime-gpu"], check=False)

        dl = SmartDownloader()
        model_checks = {}

        # Bug fix: deepinsight/inswapper (the "official" source) now
        # requires an authenticated + license-accepted Hugging Face
        # account to download (401 Unauthorized without one). Using a
        # public community mirror instead, with a second mirror as
        # fallback since these do occasionally get taken down.
        model_checks["reactor_inswapper"] = dl.download(
            "hacksider/inswapper_128", "inswapper_128.onnx",
            self.comfy / "models" / "insightface", min_size_mb=100)
        if not model_checks["reactor_inswapper"]:
            model_checks["reactor_inswapper"] = dl.download(
                "ModelsLab/inswapper", "inswapper_128.onnx",
                self.comfy / "models" / "insightface", min_size_mb=100)

        model_checks["instantid_ipadapter"] = dl.download(
            "InstantX/InstantID", "ip-adapter.bin",
            self.comfy / "models" / "instantid", min_size_mb=1000)
        controlnet_dir = self.comfy / "models" / "controlnet" / "instantid"
        model_checks["instantid_controlnet"] = dl.download(
            "InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors",
            controlnet_dir, min_size_mb=1500)

        antelope_dir = self.comfy / "models" / "insightface" / "models" / "antelopev2"
        antelope_dir.mkdir(parents=True, exist_ok=True)
        # Bug fix: DIAMONIK7777/antelopev2 via huggingface-cli download didn't
        # actually produce .onnx files (structure mismatch). Switched to
        # Aitrepreneur/insightface, which hosts the 5 required files flat
        # (no nested folder) under models/antelopev2/ — downloading each one
        # individually through the same verified SmartDownloader as
        # everything else, instead of a separate untracked huggingface-cli call.
        antelope_files = ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx",
                           "glintr100.onnx", "scrfd_10g_bnkps.onnx"]
        antelope_results = []
        for fname in antelope_files:
            antelope_results.append(dl.download(
                "Aitrepreneur/insightface", f"models/antelopev2/{fname}",
                antelope_dir, min_size_mb=1))
        antelope_ok = all(antelope_results)
        if not antelope_ok:
            self.log("⚠️ antelopev2 face-analysis models didn't fully download — this specific "
                      "set of files has historically moved between mirrors. Manual fix: open "
                      "ComfyUI-Manager's model browser inside ComfyUI and search 'antelopev2', or "
                      "download the 5 files from https://huggingface.co/Aitrepreneur/insightface/tree/main/models/antelopev2 "
                      f"and place them directly in {antelope_dir}")
        model_checks["antelopev2"] = antelope_ok

        # Bug fix: the correct SDXL PuLID file in guozinan/PuLID is
        # pulid_v1.1.safetensors — "ip-adapter_pulid_sdxl_fp16.safetensors"
        # (what this used to try) is a real filename, but it lives in a
        # DIFFERENT repo (huchenlei/ipadapter_pulid), so asking for it from
        # guozinan/PuLID 404'd every time.
        model_checks["pulid_model"] = dl.download(
            "guozinan/PuLID", "pulid_v1.1.safetensors",
            self.comfy / "models" / "pulid", min_size_mb=500)

        model_checks["ipadapter_faceid"] = dl.download(
            "h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sdxl.bin",
            self.comfy / "models" / "loras" / "ipadapter", min_size_mb=500)

        all_checks = {**clone_ok, **model_checks}
        n_ok = sum(1 for v in all_checks.values() if v)
        if n_ok == len(all_checks):
            status, detail = "ready", "ReActor + InstantID + PuLID + IPAdapter all installed with models verified."
        elif n_ok > len(all_checks) // 2:
            status, detail = "partial", f"{n_ok}/{len(all_checks)} face-tool components verified — check log above for which failed."
        else:
            status, detail = "failed", "Most face-tool installs/downloads failed — check log above."
        self._save_state("face_tools", status, all_checks, detail)

    # ------------------------------------------------------------------
    def step_upscaler(self):
        dl = SmartDownloader()
        ok = dl.download(
            "ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth",
            self.comfy / "models" / "upscale_models", min_size_mb=30)
        if not ok:
            ok = dl.download(
                "philz1337x/upscaler", "RealESRGAN_x4plus.pth",
                self.comfy / "models" / "upscale_models", min_size_mb=30)
        status = "ready" if ok else "failed"
        detail = "Real-ESRGAN model verified." if ok else "Both download sources failed — get RealESRGAN_x4plus.pth manually via ComfyUI-Manager's model browser."
        self._save_state("upscaler", status, {"model_downloaded": ok}, detail)

    # ------------------------------------------------------------------
    def step_extra_models(self):
        dl = SmartDownloader()
        checks = {}

        if _flag("AI_MEDIA_ENABLE_ZIMAGE", default="1"):
            self.log("Z-Image Turbo enabled (~19GB total) — downloading...")
            checks["zimage_text_encoder"] = dl.download(
                "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors",
                self.comfy / "models" / "text_encoders", min_size_mb=5000)
            checks["zimage_diffusion_model"] = dl.download(
                "Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                self.comfy / "models" / "diffusion_models", min_size_mb=8000)
            checks["zimage_vae"] = dl.download(
                "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors",
                self.comfy / "models" / "vae", min_size_mb=100)
        else:
            self.log("Z-Image Turbo disabled (AI_MEDIA_ENABLE_ZIMAGE=0) — skipping.")

        if _flag("AI_MEDIA_ENABLE_FLUX", default="0"):
            self.log("FLUX.1-schnell enabled (~17GB) — downloading...")
            checks["flux_checkpoint"] = dl.download(
                "black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors",
                self.comfy / "models" / "checkpoints", min_size_mb=10000)
        else:
            self.log("FLUX.1-schnell disabled (default) — set AI_MEDIA_ENABLE_FLUX=1 before this cell to enable.")

        if not checks:
            self._save_state("extra_models", "skipped", {}, "Both Z-Image Turbo and FLUX.1-schnell disabled by env vars.")
            return

        n_ok = sum(1 for v in checks.values() if v)
        status = "ready" if n_ok == len(checks) else ("partial" if n_ok else "failed")
        self._save_state("extra_models", status, checks, f"{n_ok}/{len(checks)} extra model files verified.")

    # ------------------------------------------------------------------
    def step_video(self):
        if not _flag("AI_MEDIA_ENABLE_VIDEO_NODE", default="1"):
            self._save_state("video", "skipped", {}, "Video generation node install skipped by env var.")
            return

        node_dir = self.comfy / "custom_nodes" / "ComfyUI-LTXVideo"
        ok = node_dir.exists()
        if not ok:
            r = subprocess.run(["git", "clone", "https://github.com/Lightricks/ComfyUI-LTXVideo", str(node_dir)], check=False)
            ok = r.returncode == 0
            req = node_dir / "requirements.txt"
            if req.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)

        status = "partial" if ok else "failed"
        detail = ("LTX-Video node installed, NO model downloaded yet — open ComfyUI, click the Manager "
                   "button, search 'LTX' in the model browser, and pick a checkpoint that fits your GPU's "
                   "VRAM before your first video generation.") if ok else \
                  "ComfyUI-LTXVideo node failed to clone — check your network and re-run this cell."
        self._save_state("video", status, {"node_installed": ok}, detail)

    # ------------------------------------------------------------------
    def step_start(self):
        if not self._has_gpu():
            self.log("❌ No GPU available — ComfyUI cannot start without one. "
                      "Go to Runtime > Change runtime type > T4 GPU, then Run All again.")
            self.comfy_ready = False
            self._save_state("comfyui_running", "failed", {"gpu_available": False},
                              "No GPU on this runtime — see Runtime > Change runtime type.")
            return

        comfy_log = open(self.base / "logs" / "comfyui.log", "w")

        def start():
            os.chdir(str(self.comfy))
            self.comfy_proc = subprocess.Popen(
                [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-cors-header"],
                stdout=comfy_log, stderr=subprocess.STDOUT)

        if not self._already_running("http://127.0.0.1:8188/system_stats", "ComfyUI"):
            threading.Thread(target=start, daemon=True).start()

        self.comfy_ready = self._wait_for_http("http://127.0.0.1:8188/system_stats", timeout=360, interval=3)
        if self.comfy_ready:
            self.log("ComfyUI confirmed responding on port 8188")
        else:
            self.log("⚠️ ComfyUI did not respond within 360s — check logs/comfyui.log. "
                      "It may still finish starting; a final recheck happens before tunneling.")
        self._save_state("comfyui_running", "ready" if self.comfy_ready else "partial",
                          {"responding": self.comfy_ready},
                          "ComfyUI answering HTTP." if self.comfy_ready else "Not responding yet after 360s — will recheck before tunneling.")

    # ------------------------------------------------------------------
    def step_tunnel(self):
        cf = Path("/content/cloudflared")
        try:
            if not cf.exists():
                urllib.request.urlretrieve(
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", str(cf))
                cf.chmod(0o755)
        except Exception as e:
            self.log(f"❌ Could not prepare cloudflared ({str(e)[:120]}) — no public link this session.")
            self._save_state("tunnel", "failed", {}, f"cloudflared setup failed: {str(e)[:150]}")
            return

        ready = self._wait_for_http("http://127.0.0.1:8188/system_stats", timeout=300, interval=3)
        self.public_url = None
        if ready:
            self.public_url, _ = self._start_tunnel(cf, 8188)
            self.log(f"ComfyUI URL = {self.public_url}")
        else:
            self.log("⚠️ Skipped tunnel — ComfyUI still not responding after a final 300s recheck. Check logs/comfyui.log.")

        try:
            with open(self.base / "logs" / "complete.json", "w") as f:
                json.dump({
                    "comfyui_url": self.public_url,
                    "component_status": self.state.all_status() if self.state else {},
                }, f, indent=2)
        except Exception:
            pass

        if self.public_url:
            self.log("🔓 SECURITY: no authentication on this tunnel URL — anyone with the link "
                      "can use your GPU. Don't share it, and don't leave the session open longer than needed.")
        self._save_state("tunnel", "ready" if self.public_url else "failed",
                          {"tunneled": bool(self.public_url)},
                          f"ComfyUI URL = {self.public_url}" if self.public_url else "Not tunneled — ComfyUI wasn't reachable.")

    # ------------------------------------------------------------------
    def step_final(self):
        self.log("Setup complete — see the final report below.")

    def show_done(self):
        print("\n" + "="*70)
        print("📋 AI MEDIA OS — FINAL REPORT")
        print("="*70)
        icons = {"ready": "🟢 READY  ", "partial": "🟡 PARTIAL", "failed": "🔴 FAILED ", "skipped": "⏭️  SKIPPED"}
        all_status = self.state.all_status() if self.state else {}
        order = ["comfyui_core", "face_tools", "upscaler", "extra_models", "video", "comfyui_running", "tunnel"]
        for cid in order:
            st = all_status.get(cid, {"status": "not_installed", "detail": "step did not run"})
            print(f"  {cid:16} {icons.get(st['status'], '⚪ N/A     ')}  {st.get('detail', '')[:85]}")

        print("\n" + "="*70)
        if getattr(self, "public_url", None):
            print(f"🔗 ComfyUI: {self.public_url}")
        else:
            print("🔗 ComfyUI: not reachable — check logs/comfyui.log")
            if not self._has_gpu():
                print("   Most likely cause: no GPU runtime. Runtime > Change runtime type > T4 GPU, then Run All again.")
        print(f"📁 Drive: {self.base}")
        print("="*70)


if __name__ == "__main__":
    MediaInstaller().run()
