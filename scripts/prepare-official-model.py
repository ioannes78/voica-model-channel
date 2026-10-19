#!/usr/bin/env python3
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

MODELS = {
    "punctuation": {
        "modelId": "ct-transformer-zh-en-int8",
        "kind": "PUNCTUATION",
        "displayName": "CT-Transformer zh-en punctuation int8",
        "version": "2024-04-12",
        "revision": 1,
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
        "size": 64717756,
        "sha": "c0d5aa5f8eeb686032345e180bedf39319dc2e0556781c6264bcadba8328a6e1",
        "root": "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        "files": [("model.int8.onnx", "model.int8.onnx")],
        "languages": ["zh", "en"],
        "capabilities": {},
        "attribution": "k2-fsa CT-Transformer punctuation zh-en 2024-04-12",
    },
    "sensevoice": {
        "modelId": "sensevoice-2024-int8",
        "kind": "ASR_SECOND_PASS",
        "displayName": "SenseVoice zh-en-ja-ko-yue int8",
        "version": "2024-07-17",
        "revision": 1,
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "size": 163002883,
        "sha": "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
        "root": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        "files": [("model.int8.onnx", "model.int8.onnx"), ("tokens.txt", "tokens.txt")],
        "languages": ["zh", "en", "ja", "ko", "yue"],
        "capabilities": {
            "supportsTokenTiming": True,
            "supportsLanguageDetection": True,
            "supportsInverseTextNormalization": True,
            "supportsSecondPass": True,
        },
        "attribution": "k2-fsa SenseVoice int8 2024-07-17",
    },
    "funasr-nano": {
        "modelId": "funasr-nano-int8",
        "kind": "ASR_LARGE",
        "displayName": "Fun-ASR-Nano INT8",
        "version": "2025-12-30",
        "revision": 1,
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2",
        "size": 841730611,
        "sha": "eb43d7ccc2e86b243f6a03b7df361033dda66db9523d1a92bf6aca2b50c9476b",
        "root": "sherpa-onnx-funasr-nano-int8-2025-12-30",
        "files": [
            ("encoder_adaptor.int8.onnx", "encoder_adaptor.int8.onnx"),
            ("llm.int8.onnx", "llm.int8.onnx"),
            ("embedding.int8.onnx", "embedding.int8.onnx"),
            ("Qwen3-0.6B/merges.txt", "Qwen3-0.6B/merges.txt"),
            ("Qwen3-0.6B/tokenizer.json", "Qwen3-0.6B/tokenizer.json"),
            ("Qwen3-0.6B/vocab.json", "Qwen3-0.6B/vocab.json"),
        ],
        "languages": ["zh", "en", "ja"],
        "capabilities": {
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsInverseTextNormalization": True,
            "supportsSecondPass": True,
            "supportsHotwords": True,
            "executionMode": "OFFLINE",
            "timestampCapability": "NONE",
            "supportsLanguageForcing": True,
            "punctuationMode": "NATIVE",
            "supportedParameters": [
                "numThreads",
                "maxNewTokens",
                "temperature",
                "topP",
                "seed",
                "language",
                "itn",
                "hotwords",
            ],
        },
        "runtimeModelType": "funasr-nano",
        "quantization": "INT8",
        "appVersionMin": 92,
        "licenseId": "Apache-2.0",
        "licenseUrl": "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
        "homepage": "https://k2-fsa.github.io/sherpa/onnx/funasr-nano/pretrained.html",
        "attribution": "FunAudioLLM Fun-ASR-Nano-2512; k2-fsa sherpa-onnx INT8 conversion 2025-12-30",
    },
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if len(sys.argv) != 4:
    raise SystemExit(
        "usage: prepare-official-model.py punctuation|sensevoice|funasr-nano REVISION OUT_DIR"
    )
key, revision_text, out_text = sys.argv[1:]
if key not in MODELS:
    raise SystemExit("unknown model: " + key)
revision = int(revision_text)
if revision < 1:
    raise SystemExit("revision must be >= 1")
cfg = MODELS[key]
out = pathlib.Path(out_text)
out.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as temp_text:
    temp = pathlib.Path(temp_text)
    archive = temp / "source.tar.bz2"
    with urllib.request.urlopen(cfg["url"], timeout=60) as response, archive.open("wb") as dst:
        shutil.copyfileobj(response, dst, 1024 * 1024)
    if archive.stat().st_size != cfg["size"]:
        raise SystemExit("upstream package size mismatch")
    if sha256(archive) != cfg["sha"]:
        raise SystemExit("upstream package SHA-256 mismatch")

    descriptors = []
    with tarfile.open(archive, "r:bz2") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for installed, source_name in cfg["files"]:
            package_path = cfg["root"] + "/" + source_name
            member = members.get(package_path)
            if member is None:
                raise SystemExit("missing package path: " + package_path)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise SystemExit("cannot extract: " + package_path)
            h = hashlib.sha256()
            size = 0
            while True:
                chunk = extracted.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
            descriptors.append({
                "relativePath": installed,
                "sizeBytes": size,
                "sha256": h.hexdigest(),
                "packagePath": package_path,
            })

caps = {
    "supportsStreaming": False,
    "supportsPartial": False,
    "supportsTokenTiming": False,
    "supportsLanguageDetection": False,
    "supportsConfidence": False,
    "supportsInverseTextNormalization": False,
    "supportsSecondPass": False,
    "supportsHotwords": False,
}
caps.update(cfg["capabilities"])

descriptor = {
    "modelId": cfg["modelId"],
    "kind": cfg["kind"],
    "displayName": cfg["displayName"],
    "version": cfg["version"],
    "revision": revision,
    "runtimeId": "sherpa-onnx",
    "runtimeVersionMin": "1.13.8",
    "runtimeVersionMax": None,
    "languages": cfg["languages"],
    "capabilities": caps,
    "sourceType": "MANAGED_DOWNLOAD",
    "builtinAssetPath": None,
    "download": {
        "packageFormat": "TAR_BZ2",
        "url": cfg["url"],
        "mirrors": [],
        "sizeBytes": cfg["size"],
        "sha256": cfg["sha"],
    },
    "installedSizeBytes": sum(item["sizeBytes"] for item in descriptors),
    "files": descriptors,
    "compatibility": {
        "abis": ["arm64-v8a"],
        "minSdk": 26,
        "appVersionMin": cfg.get("appVersionMin", 20),
        "appVersionMax": None,
    },
    "license": {
        "id": cfg.get("licenseId", "Apache-2.0"),
        "url": cfg.get(
            "licenseUrl",
            "https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE",
        ),
        "attribution": cfg["attribution"],
        "redistributionPolicy": "UPSTREAM_ONLY",
    },
    "sourceUrl": cfg["url"],
    "homepage": cfg.get(
        "homepage",
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html",
    ),
    "releaseChannel": "production",
    "autoUpdateEligible": False,
    "deprecated": False,
    "criticalUpdate": False,
}
if cfg.get("runtimeModelType"):
    descriptor["runtimeModelType"] = cfg["runtimeModelType"]
if cfg.get("quantization"):
    descriptor["quantization"] = cfg["quantization"]

(out / "model-descriptor.json").write_text(
    json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(descriptor, ensure_ascii=False, indent=2))
