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
        "key": "segmentation",
        "modelId": "pyannote-segmentation-3-int8",
        "kind": "SPEAKER",
        "speakerRole": "DIARIZATION_SEGMENTATION",
        "displayName": "Pyannote Segmentation 3.0 int8",
        "version": "3.0",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "packageFormat": "TAR_BZ2",
        "expectedSize": 6958444,
        "expectedSha256": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
        "root": "sherpa-onnx-pyannote-segmentation-3-0",
        "files": [("model.int8.onnx", "model.int8.onnx")],
        "languages": ["und"],
        "license": "MIT",
        "licenseUrl": "https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE",
        "attribution": "pyannote segmentation 3.0; sherpa-onnx ONNX/int8 export",
        "sourceUrl": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models",
    },
    {
        "key": "embedding",
        "modelId": "3dspeaker-eres2net-base-zh-cn-16k",
        "kind": "SPEAKER",
        "speakerRole": "EMBEDDING",
        "displayName": "3D-Speaker ERes2Net Base zh-cn 16k",
        "version": "2023-12-08",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "packageFormat": "SINGLE_FILE",
        "expectedSize": 39593761,
        "expectedSha256": "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b",
        "files": [
            (
                "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            )
        ],
        "languages": ["zh"],
        "license": "Apache-2.0",
        "licenseUrl": "https://github.com/alibaba-damo-academy/3D-Speaker/blob/main/LICENSE",
        "attribution": "3D-Speaker ERes2Net Base zh-cn 16 kHz; sherpa-onnx release asset",
        "sourceUrl": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models",
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
        headers={"User-Agent": "voica-model-channel-stage9"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as dst:
        shutil.copyfileobj(response, dst, 1024 * 1024)

def inspect_files(cfg, package: pathlib.Path):
    descriptors = []
    if cfg["packageFormat"] == "SINGLE_FILE":
        installed, source_name = cfg["files"][0]
        descriptors.append({
            "relativePath": installed,
            "sizeBytes": package.stat().st_size,
            "sha256": sha256(package),
            "packagePath": source_name,
        })
        return descriptors

    with tarfile.open(package, "r:bz2") as tar:
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
    return descriptors

if len(sys.argv) != 3:
    raise SystemExit("usage: prepare-speaker-diarization.py REVISION OUT_DIR")
revision = int(sys.argv[1])
if revision < 1:
    raise SystemExit("revision must be >= 1")
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

caps = {
    "supportsStreaming": False,
    "supportsPartial": False,
    "supportsTokenTiming": False,
    "supportsLanguageDetection": False,
    "supportsConfidence": True,
    "supportsInverseTextNormalization": False,
    "supportsSecondPass": False,
    "supportsHotwords": False,
}

descriptors = []
with tempfile.TemporaryDirectory() as temp_text:
    temp = pathlib.Path(temp_text)
    for cfg in MODELS:
        suffix = ".tar.bz2" if cfg["packageFormat"] == "TAR_BZ2" else ".onnx"
        package = temp / (cfg["key"] + suffix)
        download(cfg["url"], package)
        if package.stat().st_size != cfg["expectedSize"]:
            raise SystemExit(
                cfg["modelId"] + " upstream package size mismatch: "
                + str(package.stat().st_size)
            )
        package_sha = sha256(package)
        if package_sha != cfg["expectedSha256"]:
            raise SystemExit(
                cfg["modelId"] + " upstream package SHA-256 mismatch: " + package_sha
            )
        files = inspect_files(cfg, package)
        descriptor = {
            "modelId": cfg["modelId"],
            "kind": cfg["kind"],
            "speakerRole": cfg["speakerRole"],
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
                "packageFormat": cfg["packageFormat"],
                "url": cfg["url"],
                "mirrors": [],
                "sizeBytes": package.stat().st_size,
                "sha256": package_sha,
            },
            "installedSizeBytes": sum(item["sizeBytes"] for item in files),
            "files": files,
            "compatibility": {
                "abis": ["arm64-v8a"],
                "minSdk": 26,
                "appVersionMin": 24,
                "appVersionMax": None,
            },
            "license": {
                "id": cfg["license"],
                "url": cfg["licenseUrl"],
                "attribution": cfg["attribution"],
                "redistributionPolicy": "UPSTREAM_ONLY",
            },
            "sourceUrl": cfg["sourceUrl"],
            "homepage": "https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html",
            "releaseChannel": "production",
            "autoUpdateEligible": False,
            "deprecated": False,
            "criticalUpdate": False,
        }
        descriptors.append(descriptor)
        (out / (cfg["key"] + "-model-descriptor.json")).write_text(
            json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

(out / "speaker-model-descriptors.json").write_text(
    json.dumps(descriptors, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(descriptors, ensure_ascii=False, indent=2))
