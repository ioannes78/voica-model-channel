#!/usr/bin/env python3
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

MODELS = [
    {
        "key": "firered",
        "modelId": "fireredasr2-ctc-zh-en-int8",
        "kind": "ASR_LARGE",
        "displayName": "FireRedASR2 CTC zh-en INT8",
        "version": "2026-02-25",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2",
        "packageFormat": "TAR_BZ2",
        "expectedSize": 520516278,
        "expectedSha256": "1da8b737ecc5e29f36759a4460c754863e7c919a4ba325aea187331fbfc83274",
        "root": "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
        "explicitFiles": ["model.int8.onnx", "tokens.txt"],
        "includePrefix": None,
        "runtimeModelType": "fire-red-asr2-ctc",
        "languages": ["zh", "en"],
        "capabilities": {
            "supportsStreaming": False,
            "supportsPartial": False,
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsConfidence": False,
            "supportsInverseTextNormalization": False,
            "supportsSecondPass": True,
            "supportsHotwords": False,
            "executionMode": "OFFLINE",
            "timestampCapability": "NONE",
            "supportsLanguageForcing": False,
            "punctuationMode": "EXTERNAL",
            "supportedParameters": ["numThreads"],
        },
        "quantization": "INT8",
        "license": "Apache-2.0",
        "licenseUrl": "https://github.com/FireRedTeam/FireRedASR2S/blob/main/LICENSE",
        "attribution": "FireRedASR2 CTC zh-en INT8 2026-02-25; sherpa-onnx export",
        "homepage": "https://github.com/FireRedTeam/FireRedASR2S",
    },
    {
        "key": "qwen3",
        "modelId": "qwen3-asr-0.6b-int8",
        "kind": "ASR_LARGE",
        "displayName": "Qwen3-ASR 0.6B INT8",
        "version": "2026-03-25",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        "packageFormat": "TAR_BZ2",
        "expectedSize": 878702423,
        "expectedSha256": "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96",
        "root": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        "explicitFiles": [
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
        ],
        "includePrefix": "tokenizer/",
        "runtimeModelType": "qwen3-asr",
        "languages": ["multilingual"],
        "capabilities": {
            "supportsStreaming": False,
            "supportsPartial": False,
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsConfidence": False,
            "supportsInverseTextNormalization": False,
            "supportsSecondPass": True,
            "supportsHotwords": True,
            "executionMode": "OFFLINE",
            "timestampCapability": "NONE",
            "supportsLanguageForcing": False,
            "punctuationMode": "NATIVE",
            "supportedParameters": [
                "numThreads",
                "maxTotalLen",
                "maxNewTokens",
                "temperature",
                "topP",
                "seed",
                "hotwords",
            ],
        },
        "quantization": "INT8",
        "license": "Apache-2.0",
        "licenseUrl": "https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE",
        "attribution": "Qwen3-ASR 0.6B INT8 2026-03-25; sherpa-onnx export",
        "homepage": "https://github.com/QwenLM/Qwen3-ASR",
    },
    {
        "key": "campplus",
        "modelId": "3dspeaker-campplus-zh-cn-16k",
        "kind": "SPEAKER",
        "speakerRole": "EMBEDDING",
        "displayName": "3D-Speaker CAM++ zh-cn 16k",
        "version": "2024-10-14",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "packageFormat": "SINGLE_FILE",
        "expectedSize": 28281138,
        "expectedSha256": "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11",
        "runtimeModelType": None,
        "languages": ["zh"],
        "capabilities": {
            "supportsStreaming": False,
            "supportsPartial": False,
            "supportsTokenTiming": False,
            "supportsLanguageDetection": False,
            "supportsConfidence": True,
            "supportsInverseTextNormalization": False,
            "supportsSecondPass": False,
            "supportsHotwords": False,
        },
        "quantization": None,
        "license": "Apache-2.0",
        "licenseUrl": "https://github.com/modelscope/3D-Speaker/blob/main/LICENSE",
        "attribution": "3D-Speaker CAM++ zh-cn 16 kHz; sherpa-onnx release asset",
        "homepage": "https://github.com/modelscope/3D-Speaker",
    },
]

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, target: pathlib.Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voica-model-channel-stage13a-offline"},
    )
    with urllib.request.urlopen(request, timeout=240) as response, target.open("wb") as dst:
        shutil.copyfileobj(response, dst, 1024 * 1024)

def hash_tar_member(tar, member):
    extracted = tar.extractfile(member)
    if extracted is None:
        raise SystemExit("cannot extract: " + member.name)
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = extracted.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        h.update(chunk)
    return size, h.hexdigest()

def inspect_tar(cfg, package):
    root_prefix = cfg["root"] + "/"
    with tarfile.open(package, "r:bz2") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        selected = []
        for relative in cfg["explicitFiles"]:
            package_path = root_prefix + relative
            member = members.get(package_path)
            if member is None:
                raise SystemExit(cfg["modelId"] + " missing package path: " + package_path)
            selected.append((relative, member))
        prefix = cfg.get("includePrefix")
        if prefix:
            full_prefix = root_prefix + prefix
            extras = sorted(
                (
                    name[len(root_prefix):],
                    member,
                )
                for name, member in members.items()
                if name.startswith(full_prefix)
            )
            if not extras:
                raise SystemExit(cfg["modelId"] + " contains no files under " + prefix)
            selected.extend(extras)

        seen = set()
        descriptors = []
        for relative, member in selected:
            if relative in seen:
                continue
            seen.add(relative)
            size, digest = hash_tar_member(tar, member)
            descriptors.append({
                "relativePath": relative,
                "sizeBytes": size,
                "sha256": digest,
                "packagePath": member.name,
            })
        return descriptors

def inspect_single(cfg, package):
    name = pathlib.PurePosixPath(cfg["url"]).name
    return [{
        "relativePath": name,
        "sizeBytes": package.stat().st_size,
        "sha256": sha256(package),
        "packagePath": name,
    }]

if len(sys.argv) != 3:
    raise SystemExit("usage: prepare-stage13a-offline-speaker.py REVISION OUT_DIR")
revision = int(sys.argv[1])
if revision < 1:
    raise SystemExit("revision must be >= 1")
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

descriptors = []
with tempfile.TemporaryDirectory() as temp_text:
    temp = pathlib.Path(temp_text)
    for cfg in MODELS:
        suffix = ".tar.bz2" if cfg["packageFormat"] == "TAR_BZ2" else ".onnx"
        package = temp / (cfg["key"] + suffix)
        download(cfg["url"], package)

        actual_size = package.stat().st_size
        if actual_size != cfg["expectedSize"]:
            raise SystemExit(
                cfg["modelId"] + " upstream package size mismatch: " + str(actual_size)
            )
        package_sha = sha256(package)
        expected_sha = cfg.get("expectedSha256")
        if expected_sha is not None and package_sha != expected_sha:
            raise SystemExit(
                cfg["modelId"] + " upstream package SHA-256 mismatch: " + package_sha
            )

        files = (
            inspect_tar(cfg, package)
            if cfg["packageFormat"] == "TAR_BZ2"
            else inspect_single(cfg, package)
        )
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
            "capabilities": cfg["capabilities"],
            "sourceType": "MANAGED_DOWNLOAD",
            "builtinAssetPath": None,
            "download": {
                "packageFormat": cfg["packageFormat"],
                "url": cfg["url"],
                "mirrors": [],
                "sizeBytes": actual_size,
                "sha256": package_sha,
            },
            "installedSizeBytes": sum(item["sizeBytes"] for item in files),
            "files": files,
            "compatibility": {
                "abis": ["arm64-v8a"],
                "minSdk": 26,
                "appVersionMin": 41,
                "appVersionMax": None,
            },
            "license": {
                "id": cfg["license"],
                "url": cfg["licenseUrl"],
                "attribution": cfg["attribution"],
                "redistributionPolicy": "UPSTREAM_ONLY",
            },
            "sourceUrl": cfg["url"],
            "homepage": cfg["homepage"],
            "releaseChannel": "production",
            "autoUpdateEligible": False,
            "deprecated": False,
            "criticalUpdate": False,
        }
        if cfg.get("speakerRole"):
            descriptor["speakerRole"] = cfg["speakerRole"]
        if cfg.get("runtimeModelType"):
            descriptor["runtimeModelType"] = cfg["runtimeModelType"]
        if cfg.get("quantization"):
            descriptor["quantization"] = cfg["quantization"]

        descriptors.append(descriptor)
        (out / (cfg["key"] + "-model-descriptor.json")).write_text(
            json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

(out / "offline-speaker-model-descriptors.json").write_text(
    json.dumps(descriptors, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(descriptors, ensure_ascii=False, indent=2))
