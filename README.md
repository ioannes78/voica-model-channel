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


## Candidate workflows

- **Prepare Small Bilingual Candidate**: builds the generic Android CPU package
  from the pinned upstream 2023-02-16 archive, publishes the model ZIP plus the
  candidate `production.json` as a GitHub prerelease, and opens a human-gated
  manifest PR.
- **Prepare Official Upstream Candidate**: verifies the pinned official package
  SHA-256 for CT-Transformer punctuation or SenseVoice, publishes the candidate
  `production.json` as a GitHub prerelease, and opens a human-gated manifest PR
  that still points the model download at the official upstream archive.
- **Validate Production Manifest**: recomputes the models-array digest, checks
  HTTPS/hash/path invariants, and prevents removal/change of the APK Silero baseline.

Merging a candidate PR is the publication action. Candidate generation alone never
changes the App-visible production manifest.

## Pre-merge Android validation

A Debug Voica APK can temporarily use the candidate manifest release asset before
the candidate PR is merged. The accepted URL shape is:

`https://github.com/ioannes78/voica-model-channel/releases/download/<candidate-tag>/production.json`

Use that URL only in the Debug-only Stage 8 candidate model control. After saving,
fully exit and reopen Voica so its ModelManager is recreated from the candidate
catalog. Download and activate the candidate, complete the native sherpa smoke test
and the relevant transcription acceptance, then merge the manifest PR. Restore the
Debug App to production afterward.

Release builds do not accept this override and always use the production manifest.

## Stage 8 candidate order and gates

Candidates are intentionally validated and merged in dependency order:

1. **Small Bilingual r1** — verify package/file integrity and pass the
   streaming-ASR native smoke activation, then merge its manifest PR. A complete
   Fast transcription cannot run yet because punctuation is not production-ready.
2. **CT-Transformer punctuation r1** — the generated candidate manifest now includes
   production Small Bilingual. Pass punctuation native smoke **and a real short Fast
   transcription** through VAD -> streaming ASR -> punctuation, then merge.
3. **SenseVoice 2024 int8 r1** — the candidate manifest now includes production
   Small Bilingual + punctuation. Pass SenseVoice native smoke **and a real short
   High Quality two-pass transcription**, then merge.

The official-candidate workflow enforces these prerequisites and fails if punctuation
is requested before Small Bilingual is in production, or SenseVoice is requested
before both Small Bilingual and punctuation are in production.
