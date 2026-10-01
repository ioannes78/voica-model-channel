# Voica Model Channel

Public, human-gated model distribution channel for
[`ioannes78/voica-android`](https://github.com/ioannes78/voica-android).

## Release policy

Voica uses a semi-automatic model publication flow:

1. A maintainer manually starts a candidate workflow.
2. CI downloads a pinned upstream model asset and verifies its expected package
   identity/size/hash where the upstream provides one.
3. CI prepares or inspects the candidate and emits package/file SHA-256 metadata.
4. The candidate is installed on a real Android device and must pass Voica's
   ModelManager integrity check plus the sherpa-onnx native smoke gate.
5. Only after real-device evidence exists is `manifests/production.json` updated.
6. The Android app never activates a newly downloaded model silently. Downloaded
   candidates require a successful native smoke test before activation.

## Production manifest

The Android app reads:

`https://raw.githubusercontent.com/ioannes78/voica-model-channel/main/manifests/production.json`

The initial production manifest contains only the built-in Silero VAD baseline so
that update checks are valid before downloadable ASR/punctuation packages are
promoted.

## Stage 8 first model set

- VAD: Silero VAD int8 — built into the APK, managed override supported.
- Fast/streaming ASR: Small Bilingual Zipformer zh-en 2023-02-16, generic CPU
  package prepared from the documented int8 encoder + fp32 decoder + int8 joiner
  + tokens layout.
- Punctuation: CT-Transformer zh-en int8 2024-04-12.
- High-quality second pass: SenseVoice zh/en/ja/ko/yue int8 2024-07-17.

Do not use the RK356x/RK3576/RK3588 Small Bilingual packages as the generic Android
arm64 CPU package; those assets target Rockchip acceleration.
