#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE_PACKAGE_NAME = "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2"
BASE_PACKAGE_SIZE = 841_730_611
BASE_PACKAGE_SHA256 = "eb43d7ccc2e86b243f6a03b7df361033dda66db9523d1a92bf6aca2b50c9476b"
BASE_ROOT = "sherpa-onnx-funasr-nano-int8-2025-12-30"
CANDIDATE_ROOT = "voica-funasr-nano-int8-2025-12-30-r2"
CANDIDATE_PACKAGE_NAME = CANDIDATE_ROOT + ".tar.bz2"
DUAL_ENCODER_SHA256 = "d5bb172a133fc4f996ede12cab43c0b9b4bcbb2922c1f967794505b9549b6d85"
DUAL_ENCODER_SIZE = 238_277_774
CTC_SCORER_SHA256 = "db93af357c1d5ba78e9516950fd8931ead11666ffa00b1928e2fda9e8003bc38"
CTC_SCORER_SIZE = 54_235_091
TOKENIZER_SHA256 = "747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718"
EXPECTED_BLANK_ID = 60_514
EXPECTED_VOCAB_SIZE = 60_515
EXPECTED_FRAME_PERIOD_US = 60_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, *, size: int | None = None, sha256: str | None = None) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"Size mismatch for {path}: {path.stat().st_size} != {size}")
    if sha256 is not None:
        actual = sha256_file(path)
        if actual != sha256:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {sha256}")
    return path


def safe_extract(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(package, "r:bz2") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def locate_single(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}, got {matches}")
    return matches[0]


def file_entry(root: Path, relative_path: str) -> dict[str, object]:
    path = require_file(root / relative_path)
    return {
        "relativePath": relative_path,
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "packagePath": f"{CANDIDATE_ROOT}/{relative_path}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2a-root", type=Path, required=True)
    parser.add_argument("--h2c-root", type=Path, required=True)
    parser.add_argument("--h2d-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()

    h2a_root = args.h2a_root.resolve()
    h2c_root = args.h2c_root.resolve()
    h2d_root = args.h2d_root.resolve()
    dist = args.dist.resolve()
    dist.mkdir(parents=True, exist_ok=True)

    base_package = require_file(
        h2c_root / BASE_PACKAGE_NAME,
        size=BASE_PACKAGE_SIZE,
        sha256=BASE_PACKAGE_SHA256,
    )
    dual_encoder = require_file(
        h2c_root / "encoder_adaptor.dual.int8.onnx",
        size=DUAL_ENCODER_SIZE,
        sha256=DUAL_ENCODER_SHA256,
    )
    ctc_scorer = require_file(
        h2d_root / "ctc_compact.int8.onnx",
        size=CTC_SCORER_SIZE,
        sha256=CTC_SCORER_SHA256,
    )
    tokenizer = locate_single(h2a_root, "multilingual.tiktoken")
    require_file(tokenizer, sha256=TOKENIZER_SHA256)

    work = dist / "work"
    if work.exists():
        shutil.rmtree(work)
    safe_extract(base_package, work)
    base_root = work / BASE_ROOT
    if not base_root.is_dir():
        raise RuntimeError(f"Unexpected upstream package root: {base_root}")
    candidate_root = work / CANDIDATE_ROOT
    base_root.rename(candidate_root)

    shutil.copy2(dual_encoder, candidate_root / "encoder_adaptor.int8.onnx")
    shutil.copy2(ctc_scorer, candidate_root / "ctc_compact.int8.onnx")
    shutil.copy2(tokenizer, candidate_root / "multilingual.tiktoken")

    metadata = "\n".join(
        [
            "schemaVersion=1",
            "scoreDomain=raw_logits",
            "encoderPostAdaptorOutputIndex=0",
            "encoderPostAdaptorOutputName=encoder_out",
            "encoderPreAdaptorOutputIndex=1",
            "encoderPreAdaptorOutputName=ctc_encoder_out",
            "encoderPreAdaptorLengthOutputIndex=2",
            "encoderPreAdaptorLengthOutputName=ctc_encoder_out_lens",
            "encoderPreAdaptorDim=512",
            "ctcEncoderInputName=ctc_encoder_out",
            "ctcEncoderLengthInputName=ctc_encoder_out_lens",
            "ctcCandidateIdsInputName=candidate_ids",
            "ctcCandidateScoresOutputName=candidate_logits",
            f"ctcVocabSize={EXPECTED_VOCAB_SIZE}",
            f"blankId={EXPECTED_BLANK_ID}",
            f"framePeriodUs={EXPECTED_FRAME_PERIOD_US}",
            "tokenizerFormat=sensevoice_tiktoken",
            "tokenizerFile=multilingual.tiktoken",
            "",
        ]
    )
    (candidate_root / "ctc-alignment.properties").write_text(metadata, encoding="utf-8")

    relative_paths = [
        "encoder_adaptor.int8.onnx",
        "llm.int8.onnx",
        "embedding.int8.onnx",
        "Qwen3-0.6B/merges.txt",
        "Qwen3-0.6B/tokenizer.json",
        "Qwen3-0.6B/vocab.json",
        "ctc_compact.int8.onnx",
        "multilingual.tiktoken",
        "ctc-alignment.properties",
    ]
    files = [file_entry(candidate_root, relative_path) for relative_path in relative_paths]

    package = dist / CANDIDATE_PACKAGE_NAME
    package.unlink(missing_ok=True)
    subprocess.run(
        ["tar", "-C", str(work), "-cjf", str(package), CANDIDATE_ROOT],
        check=True,
    )
    package_size = package.stat().st_size
    package_sha = sha256_file(package)

    catalog = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    matches = [item for item in catalog["models"] if item.get("modelId") == "funasr-nano-int8"]
    if len(matches) != 1:
        raise RuntimeError(f"Base manifest must contain one FunASR descriptor, got {len(matches)}")
    descriptor = copy.deepcopy(matches[0])
    descriptor["revision"] = 2
    descriptor["displayName"] = "Fun-ASR-Nano INT8 + native CTC (P5 QA)"
    descriptor["capabilities"]["supportsTokenTiming"] = True
    descriptor["capabilities"]["timestampCapability"] = "TOKEN"
    descriptor["download"] = {
        "packageFormat": "TAR_BZ2",
        "url": (
            "https://github.com/ioannes78/voica-model-channel/releases/download/"
            + args.release_tag
            + "/"
            + CANDIDATE_PACKAGE_NAME
        ),
        "mirrors": [],
        "sizeBytes": package_size,
        "sha256": package_sha,
    }
    descriptor["installedSizeBytes"] = sum(int(item["sizeBytes"]) for item in files)
    descriptor["files"] = files
    descriptor["compatibility"]["appVersionMin"] = 92
    descriptor["license"]["attribution"] = (
        "FunAudioLLM Fun-ASR-Nano-2512; k2-fsa sherpa-onnx INT8 base; "
        "Voica Stage 16B P5 QA dual-output encoder + compact CTC scorer"
    )
    descriptor["license"]["redistributionPolicy"] = "VOICA_MIRROR_ALLOWED"
    descriptor["sourceUrl"] = "https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512"
    descriptor["releaseChannel"] = "candidate"
    descriptor["autoUpdateEligible"] = False
    descriptor["deprecated"] = False
    descriptor["criticalUpdate"] = False
    descriptor["runtimeModelType"] = "funasr-nano"
    descriptor["quantization"] = "INT8"

    descriptor_path = dist / "model-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    package_metadata = {
        "candidate": "Stage 16B P5 FunASR native CTC revision 2",
        "releaseTag": args.release_tag,
        "basePackage": {
            "name": BASE_PACKAGE_NAME,
            "sizeBytes": BASE_PACKAGE_SIZE,
            "sha256": BASE_PACKAGE_SHA256,
        },
        "dualEncoder": {
            "sizeBytes": DUAL_ENCODER_SIZE,
            "sha256": DUAL_ENCODER_SHA256,
        },
        "ctcCompact": {
            "sizeBytes": CTC_SCORER_SIZE,
            "sha256": CTC_SCORER_SHA256,
        },
        "multilingualTiktoken": {
            "sizeBytes": tokenizer.stat().st_size,
            "sha256": TOKENIZER_SHA256,
        },
        "package": {
            "name": CANDIDATE_PACKAGE_NAME,
            "sizeBytes": package_size,
            "sha256": package_sha,
        },
        "ctcContract": {
            "blankId": EXPECTED_BLANK_ID,
            "vocabSize": EXPECTED_VOCAB_SIZE,
            "framePeriodUs": EXPECTED_FRAME_PERIOD_US,
        },
    }
    (dist / "package-metadata.json").write_text(
        json.dumps(package_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(package_metadata, ensure_ascii=False, indent=2))
    print(f"PACKAGE_PATH={package}")
    print(f"DESCRIPTOR_PATH={descriptor_path}")


if __name__ == "__main__":
    main()
