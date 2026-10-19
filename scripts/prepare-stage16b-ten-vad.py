#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys
import urllib.request

MODEL_ID = "ten-vad-fp32"
VERSION = "2025-07-11"
REVISION = 1
RUNTIME_ID = "sherpa-onnx"
RUNTIME_VERSION = "1.13.8"
RELEASE_TAG = "candidate-ten-vad-r1"
MODEL_NAME = "ten-vad.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/ten-vad.onnx"
)
MODEL_SIZE = 332_211
MODEL_SHA256 = "718cb7eef47e3cf5ddbe7e967a7503f46b8b469c0706872f494dfa921b486206"
LICENSE_URL = "https://github.com/TEN-framework/ten-vad/blob/main/LICENSE"
HOMEPAGE = "https://github.com/TEN-framework/ten-vad"


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voica-model-channel-stage16b/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare-stage16b-ten-vad.py OUT_DIR")

    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    data = download(MODEL_URL)
    if len(data) != MODEL_SIZE:
        raise SystemExit(
            f"upstream size mismatch: expected {MODEL_SIZE}, got {len(data)}"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(
            f"upstream SHA-256 mismatch: expected {MODEL_SHA256}, got {digest}"
        )

    descriptor = {
        "modelId": MODEL_ID,
        "kind": "VAD",
        "displayName": "TEN VAD FP32",
        "version": VERSION,
        "revision": REVISION,
        "runtimeId": RUNTIME_ID,
        "runtimeVersionMin": RUNTIME_VERSION,
        "runtimeVersionMax": RUNTIME_VERSION,
        "languages": ["und"],
        "capabilities": {
            "supportsStreaming": True,
            "supportsPartial": False,
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsConfidence": False,
            "supportsInverseTextNormalization": False,
            "supportsSecondPass": False,
            "supportsHotwords": False,
        },
        "sourceType": "MANAGED_DOWNLOAD",
        "builtinAssetPath": None,
        "download": {
            "packageFormat": "SINGLE_FILE",
            "url": MODEL_URL,
            "mirrors": [],
            "sizeBytes": MODEL_SIZE,
            "sha256": MODEL_SHA256,
        },
        "installedSizeBytes": MODEL_SIZE,
        "files": [
            {
                "relativePath": MODEL_NAME,
                "sizeBytes": MODEL_SIZE,
                "sha256": MODEL_SHA256,
                "packagePath": MODEL_NAME,
            }
        ],
        "compatibility": {
            "abis": ["arm64-v8a"],
            "minSdk": 26,
            "appVersionMin": 92,
            "appVersionMax": None,
        },
        "license": {
            "id": "TEN-VAD-Open-Source-License",
            "url": LICENSE_URL,
            "attribution": (
                "TEN VAD; sherpa-onnx official release asset. License is Apache-2.0 "
                "with TEN-VAD additional deployment conditions; production adoption "
                "requires a separate license applicability review."
            ),
            "redistributionPolicy": "UPSTREAM_ONLY",
        },
        "sourceUrl": MODEL_URL,
        "homepage": HOMEPAGE,
        "releaseChannel": "production",
        "autoUpdateEligible": False,
        "deprecated": False,
        "criticalUpdate": False,
        "quantization": "FP32",
    }

    metadata = {
        "releaseTag": RELEASE_TAG,
        "modelName": MODEL_NAME,
        "modelUrl": MODEL_URL,
        "modelSizeBytes": MODEL_SIZE,
        "modelSha256": MODEL_SHA256,
        "runtimeId": RUNTIME_ID,
        "runtimeVersion": RUNTIME_VERSION,
        "redistributionPolicy": "UPSTREAM_ONLY",
        "licenseUrl": LICENSE_URL,
    }

    (out / "model-descriptor.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "package-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(descriptor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
