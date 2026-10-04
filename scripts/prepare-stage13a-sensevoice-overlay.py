#!/usr/bin/env python3
import copy
import json
import pathlib
import sys

MODEL_ID = "sensevoice-2024-int8"

if len(sys.argv) != 3:
    raise SystemExit("usage: prepare-stage13a-sensevoice-overlay.py BASE_MANIFEST OUT_DESCRIPTOR")

base_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
manifest = json.loads(base_path.read_text(encoding="utf-8"))
matches = [m for m in manifest.get("models", []) if m.get("modelId") == MODEL_ID]
if len(matches) != 1:
    raise SystemExit("expected exactly one " + MODEL_ID + " descriptor")

descriptor = copy.deepcopy(matches[0])
capabilities = copy.deepcopy(descriptor.get("capabilities", {}))
capabilities.update({
    "supportsStreaming": False,
    "supportsPartial": False,
    "supportsTokenTiming": True,
    "supportsLanguageDetection": True,
    "supportsConfidence": False,
    "supportsInverseTextNormalization": True,
    "supportsSecondPass": True,
    "supportsHotwords": False,
    "executionMode": "OFFLINE",
    "timestampCapability": "TOKEN",
    "supportsLanguageForcing": True,
    "punctuationMode": "EXTERNAL",
    "supportedParameters": [
        "numThreads",
        "language",
        "useInverseTextNormalization",
    ],
})
descriptor["capabilities"] = capabilities
descriptor["quantization"] = descriptor.get("quantization") or "INT8"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(descriptor, ensure_ascii=False, indent=2))
