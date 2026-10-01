#!/usr/bin/env python3
import datetime
import hashlib
import json
import pathlib
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: upsert_model.py BASE_MANIFEST MODEL_DESCRIPTOR OUTPUT")

base_path = pathlib.Path(sys.argv[1])
descriptor_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])

catalog = json.loads(base_path.read_text(encoding="utf-8"))
model = json.loads(descriptor_path.read_text(encoding="utf-8"))
model_id = model["modelId"]

models = [m for m in catalog["models"] if m["modelId"] != model_id]
models.append(model)
models.sort(key=lambda item: item["modelId"])

catalog["models"] = models
catalog["manifestVersion"] = int(catalog["manifestVersion"]) + 1
catalog["publishedAt"] = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
compact = json.dumps(models, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
catalog["manifestDigest"] = hashlib.sha256(compact).hexdigest()
catalog["signature"] = None
catalog["keyId"] = None

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print("updated", model_id, "manifestDigest", catalog["manifestDigest"])
