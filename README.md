# 🎨 AI Media OS

ComfyUI, pre-loaded with a curated, verified set of free/open-source tools
for: image generation, face swap, face lock (same character every time),
background change, and upscaling. Video generation is available as an
opt-in extra.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhayraj01op/AI-MEDIA-OS/blob/main/colab/AI_Media_OS.ipynb)

## ⚠️ #1 cause of "no link at the end": GPU runtime not selected
If Step 3 finishes but ComfyUI never comes up, check the bottom-right
corner of the Colab window. It must say **"T4 (Python 3)"** — if it just
says "Python 3" with no GPU name, you're on a CPU-only runtime and
ComfyUI **cannot start at all** (it needs a CUDA-enabled GPU). Fix:
**Runtime menu → Change runtime type → T4 GPU → Save**, then **Runtime →
Run All** again. Step 5 (Debug ComfyUI) checks for this directly and
tells you if it's the problem.

## What's inside
| Capability | Tool | Notes |
|---|---|---|
| Image generation | SDXL (always) + Z-Image Turbo (default on) | SDXL is required for the face tools below; Z-Image Turbo is newer/faster/higher quality for general generation |
| Face swap | ReActor | Works on both images and video frames |
| Face lock (consistent character) | InstantID + PuLID + IPAdapter | All three are SDXL-specific — this is why SDXL is always downloaded |
| Background change | ComfyUI's built-in inpaint/outpaint nodes | No extra install — it's workflow, not a tool |
| Upscaling | Real-ESRGAN | |
| Video generation | ComfyUI-LTXVideo node (optional) | Model itself isn't auto-downloaded — see below |
| Alternative image model | FLUX.1-schnell (optional, off by default) | Turn on in Step 2 if you want it alongside/instead of Z-Image |

## ⚡ One-Click Start
1. Push this project's files to your own GitHub repo, update `REPO_URL` in
   Step 1 of the notebook if your repo name is different.
2. Colab → Runtime → Change runtime type → **T4 GPU** (this step is easy
   to forget and is the #1 reason nothing shows up at the end — see above).
3. Run Step 1, adjust Step 2's toggles if you want, then Run All (or run
   Steps 3-4 individually).
4. First run is realistically **30-60+ minutes**.
5. Step 4 gives you the ComfyUI link. If it's missing, run Step 5.

## Model sizes
- SDXL base: ~7GB (always downloaded — required by the face tools)
- Face tools (ReActor + InstantID + PuLID + IPAdapter models): ~5GB combined
- Real-ESRGAN: ~65MB
- Z-Image Turbo: ~19GB (on by default — toggle off in Step 2 if you don't want it)
- FLUX.1-schnell: ~17GB (off by default)

## About video generation
This notebook installs the **ComfyUI-LTXVideo node** but does not
auto-download an LTX-Video model checkpoint — the current model line
needs 16GB+ VRAM even for its lightest version, tighter than a free T4
has to spare, and file sizes/URLs in this space change every few months.
Once ComfyUI is running, click **Manager**, open the model browser,
search "LTX", and pick whichever current checkpoint fits your GPU.

## What persists across a Colab runtime reset, and what doesn't
Everything under Google Drive persists: ComfyUI itself, all custom nodes,
every model file, and your outputs. The VM's Python environment
(`pip install`) has to run again on a brand-new session — a Colab
constraint, not a bug here — though the Drive-cached pip downloads make
that faster.

## Security
The ComfyUI tunnel URL has **no authentication** — anyone with the link
can generate images/videos using your GPU. Don't share it, and don't
leave the session open longer than you need to.

## Face-lock quick start
1. Load an SDXL checkpoint (already downloaded).
2. Add an **InstantID** or **IPAdapter FaceID** loader node, feed it a
   reference face photo.
3. Generate — the output should keep the same face across different
   prompts/poses. Try PuLID too if InstantID's results look too "stiff".
4. For putting that face onto an *existing* image/video: use **ReActor**
   instead — that's face swap, a different workflow from face-locked
   generation.

If `antelopev2` face-analysis models didn't auto-download (this specific
file has a history of moving between mirrors), the install log tells you
exactly what to search for in ComfyUI-Manager or where to get it manually.
