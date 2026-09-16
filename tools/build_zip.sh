#!/usr/bin/env bash
# build_zip.sh — package the submission zip.
#
# Select public (blank credentials) or private (local credentials) explicitly:
#   ./tools/build_zip.sh --public bundle
#   ./tools/build_zip.sh --private bundle
#   ./tools/build_zip.sh --public hf <repo_id>
#
# Public builds produce dist/ble_public.zip; private builds produce dist/ble_submission.zip.
# Both place model.py at the ZIP ROOT (required by the platform).

set -euo pipefail
REPOSITORY_ROOT=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().parents[1])' "$0")
cd "$REPOSITORY_ROOT"

case "${1:-}" in
  --public) CONFIG=ble/submission_config.example.json; OUT=dist/ble_public.zip ;;
  --private) CONFIG=ble/submission_config.json; OUT=dist/ble_submission.zip ;;
  *) echo "usage: build_zip.sh --public|--private bundle|hf [repo_id]" >&2; exit 2 ;;
esac
RELEASE_KIND="$1"
shift
MODE="${1:-}"
if { [ "$MODE" = bundle ] && [ "$#" -ne 1 ]; } || \
   { [ "$MODE" = hf ] && [ "$#" -ne 2 ]; } || \
   { [ "$MODE" != bundle ] && [ "$MODE" != hf ]; }; then
  echo "usage: build_zip.sh --public|--private bundle|hf [repo_id]" >&2
  exit 2
fi
[ -f "$CONFIG" ] || { echo "missing configuration: $CONFIG" >&2; exit 1; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

if [ -e ble/subjects.txt ] || [ -e ble/subjects.txt.example ]; then
  echo "subjects.txt is unsupported; the platform selects evaluated subjects" >&2
  exit 1
fi

# Stage an explicit allowlist so ignored or generated files are not swept into
# the archive accidentally.
cp ble/model.py ble/labeling.py ble/requirements.txt "$STAGE"/
python3 - "$STAGE" "$CONFIG" "$RELEASE_KIND" <<'EOF'
import json, pathlib, shutil, sys
stage = pathlib.Path(sys.argv[1])
cfg = json.loads(pathlib.Path(sys.argv[2]).read_text())
if sys.argv[3] == "--public" and any(cfg.get("api_keys", {}).values()):
    raise SystemExit("public configuration must contain only blank API keys")
(stage / "submission_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
for directory in ("src", "comp_pipeline"):
    for source in (pathlib.Path("ble") / directory).rglob("*.py"):
        if source.is_symlink() or any(p.is_symlink() for p in source.parents):
            raise SystemExit("refusing to package symlinked source")
        target = stage / source.relative_to("ble")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
EOF

if [ "$MODE" = hf ]; then
  REPO="${2:?usage: build_zip.sh hf <hf_repo_id>}"
  echo "$REPO" > "$STAGE/models.txt"
  python3 - "$STAGE" "$REPO" <<'EOF'
import json, sys
p = f"{sys.argv[1]}/submission_config.json"
cfg = json.load(open(p)); cfg["hf_data_repo"] = sys.argv[2]
json.dump(cfg, open(p, "w"), indent=2)
EOF
elif [ "$MODE" = bundle ]; then
  [ -d payload/data ] || { echo "payload/data missing — run tools/prepare_data.py first"; exit 1; }
  python3 - payload/data/corpus_manifest.json <<'EOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
revision = manifest.get("revision", "")
# Preparation records the current main snapshot; no historical SHA is required.
try:
    revision_is_valid = len(revision) == 40 and int(revision, 16) >= 0
except (TypeError, ValueError):
    revision_is_valid = False
valid = (
    manifest.get("corpus_scope") == "paec-public-training"
    and manifest.get("db_repo") == "aims-foundations/measurement-db"
    and revision_is_valid
)
if not valid:
    raise SystemExit(
        "payload is not marked as the official public training corpus; "
        "rebuild it into an empty directory"
    )
EOF
  cp -r payload/data "$STAGE/data"
  [ -d payload/embeddings ] && cp -r payload/embeddings "$STAGE/embeddings"
else
  echo "unknown mode: $MODE (use 'hf <repo_id>' or 'bundle')"; exit 1
fi

if find "$STAGE" -name subjects.txt -print -quit | grep -q .; then
  echo "refusing to package unsupported subjects.txt" >&2
  exit 1
fi

# Public archives need the recipient's own credentials before a real API run.
python3 - "$STAGE" "$RELEASE_KIND" <<'EOF'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/submission_config.json"))
missing = [k for k, v in cfg.get("api_keys", {}).items() if not v]
if sys.argv[2] == "--public":
    print("Public archive: configure your own credentials before submitting.")
elif missing:
    print(f"WARNING: empty api_keys in submission_config.json: {missing}")
    print("The sandbox has no environment keys — LLM predictions will fail.")
else:
    print("Private submission archive: do not publish this file.")
EOF

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
(cd "$STAGE" && zip -qr - .) > "$OUT"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | sed -n '1,25p'
