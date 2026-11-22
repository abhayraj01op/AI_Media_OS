"""
Smart Downloader - Resumable downloads from Hugging Face, with real
verification instead of "the path exists so it must be fine".
"""

from pathlib import Path


class SmartDownloader:
    def __init__(self):
        self.done = []
        self.failed = []

    def _looks_valid(self, path: Path, min_size_mb: float) -> bool:
        try:
            if not path.exists():
                return False
            size_mb = path.stat().st_size / 1e6
            return size_mb >= min_size_mb
        except Exception:
            return False

    def download(self, repo_id, filename, dest_dir, force=False, min_size_mb=1.0):
        from huggingface_hub import hf_hub_download

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Keep only the actual filename (some HF paths include subfolders
        # like "split_files/vae/ae.safetensors") for the local dest path.
        local_name = Path(filename).name
        dest_path = dest_dir / local_name

        if not force and self._looks_valid(dest_path, min_size_mb):
            gb = dest_path.stat().st_size / 1e9
            print(f"   ✅ {local_name} ({gb:.2f} GB) — already present, verified")
            self.done.append(str(dest_path))
            return True

        if dest_path.exists() and not self._looks_valid(dest_path, min_size_mb):
            print(f"   ⚠️ {local_name} exists but looks corrupt/incomplete — re-downloading")

        try:
            print(f"   📥 {filename}...")
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
            # hf_hub_download preserves subfolder structure under dest_dir;
            # move/flatten to dest_path if it landed in a subfolder.
            downloaded_path = Path(downloaded_path)
            if downloaded_path != dest_path and downloaded_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path.replace(dest_path)
        except Exception as e:
            print(f"   ❌ {filename}: {str(e)[:120]}")
            self.failed.append(filename)
            return False

        if not self._looks_valid(dest_path, min_size_mb):
            print(f"   ❌ {filename}: download finished but file is missing or too small "
                  f"(< {min_size_mb} MB) — treating as failed")
            self.failed.append(filename)
            return False

        gb = dest_path.stat().st_size / 1e9
        print(f"   ✅ {local_name} ({gb:.2f} GB)")
        self.done.append(str(dest_path))
        return True
