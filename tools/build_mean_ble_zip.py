"""Build the empirical mean baseline with BLE uncertainty acquisition."""

from pathlib import Path

from ble_packaging import arguments, ble_files, source_file, write_archive


def build_zip(root: Path, *, public: bool, mode: str = "bundle",
              repo_id: str | None = None, output: Path | None = None) -> Path:
    baseline = root / "empirical_mean_ble_acquisition"
    files = {name: source_file(baseline, name)
             for name in ("model.py", "labeling.py", "README.md")}
    files["empirical_mean/model.py"] = source_file(root / "empirical_mean", "model.py")
    files["empirical_mean/__init__.py"] = b""
    files["ble/__init__.py"] = b""
    for name, value in ble_files(root, public=public, mode=mode, repo_id=repo_id).items():
        # Dependency declarations and prefetch instructions belong at ZIP root.
        files[name if name in {"requirements.txt", "models.txt"} else "ble/" + name] = value
    suffix = "public" if public else "submission"
    output = output or root / "dist" / f"empirical_mean_ble_acquisition_{suffix}.zip"
    return write_archive(files, output)


def main():
    args = arguments("Build empirical mean predictions with BLE acquisition")
    path = build_zip(Path(__file__).resolve().parents[1], public=args.public,
                     mode=args.mode, repo_id=args.repo_id)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
    print("Public archive: configure BLE credentials before submitting."
          if args.public else "Private submission archive: do not publish this file.")


if __name__ == "__main__":
    main()
