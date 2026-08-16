# Stage E Acquisition QC

Date: 2026-08-15
Branch baseline before this note: `5464ed5` (`research-v4.3-stage-cd4c1`)

## Status

Stage E.1 autonomous source-media acquisition is accepted for embeddable YouTube videos using the `web_embedded` client and yt-dlp's native HTTP downloader.

Stage E.2 autonomous multi-frame visual transcription is also accepted for the QC target. Seven independently extracted frames all produced the same equation structure with `qwen3-vl:8b-instruct` at temperature 0, with no uncertain tokens. No Stage E production source implementation is included in this checkpoint; this file records the accepted acquisition and visual-consensus contracts and QC evidence only.

## Historical acquisition replay

The exact bounded-download command that succeeded on 2026-08-04 was replayed on 2026-08-15 with the same project yt-dlp stable release (`2026.07.04`). It now fails with HTTP 403 when ffmpeg opens the Googlevideo URL selected as format `137+251`.

Conclusion: the earlier bounded section-download behavior is not currently reliable enough to use as the Stage E production acquisition path.

## Rejected acquisition paths

### `android_vr` / direct HTTPS

A native yt-dlp video-only download selecting format 137 failed with HTTP 403.

### `mweb` + automatic PO token + bounded remote ffmpeg

Environment used:

- yt-dlp `2026.07.04`
- Deno `2.9.4`
- `bgutil-ytdlp-pot-provider` `1.3.1`
- provider source tag `1.3.1`, commit `7608dd5`

The bgutil Deno provider was discovered successfully. yt-dlp generated and retrieved a video-bound GVS PO token for the `mweb` client and appended the resulting `pot` value to the media URL. The ffmpeg-backed `--download-sections` request still failed with HTTP 403.

### `mweb` + automatic PO token + yt-dlp native HTTP

The same `mweb` + bgutil PO-token setup was tested without `--download-sections` and without remote ffmpeg. yt-dlp's native HTTP downloader selected video-only format 137 and still received HTTP 403.

Conclusion: the observed failure is not specific to ffmpeg's remote downloader, and successful PO-token generation alone is not sufficient for this target/session.

## Accepted Stage E.1 path: `web_embedded` + native HTTP

Target video:

- Video ID: `pfhjJ00IuW4`
- Title: `8. Value a Bond and Calculate Yield to Maturity (YTM)`

Acquisition configuration:

- YouTube client: `web_embedded`
- yt-dlp: `2026.07.04`
- JavaScript runtime: Deno `2.9.4`
- downloader: yt-dlp native HTTP
- no PO token required for the successful request
- no `--download-sections`
- no remote ffmpeg request
- selected format: `137`

Accepted media evidence:

- codec: H.264
- resolution: `1920x1080`
- duration: `971.833333` seconds
- size: `24,437,770` bytes
- source-media SHA-256: `765c80fb016098cbf9741f178e8b9e478ce9621fa760a0fb274b6f4f26cc24c1`

A local ffmpeg seek against the downloaded media at 785.5 seconds succeeded.

Accepted frame evidence from the initial acquisition test:

- timestamp: `785.5` seconds
- frame SHA-256: `5261173c9e8cbb80d0c7557acf851a8fd611cb1c611c660d410dd76d8ca3c146`

## Accepted Stage E.2 multi-frame visual consensus

Cue window:

- S161: `783.630-788.630` — transcript announces the displayed equation.
- S162: `788.630-793.630` — transcript continues referring to the equation.

Seven frames were extracted independently from the autonomously acquired `web_embedded` source using local ffmpeg with `-nostdin`:

| timestamp | frame SHA-256 |
|---:|---|
| 784.0 | `29cae8ae43f01f39dabc7841bfb3314d012bdc8d3b7b506b32d058030b279e9d` |
| 785.5 | `5261173c9e8cbb80d0c7557acf851a8fd611cb1c611c660d410dd76d8ca3c146` |
| 787.0 | `2d623901f6b135798ffd7f8e7f6e0e9bd520b111f96a82473f03d124a98df7eb` |
| 788.5 | `a1d3de683c17e6b42de3361d1e5197a260306f9c2e1b7546e39c5001f05f646b` |
| 790.0 | `74518b27f8c89f2f7ac1c547b118ef296f6522db04913afee49236920547593d` |
| 791.5 | `275cef30a205e42085bbe69df513188d1d449de7f908b4aaf2da524f685f9631` |
| 793.0 | `76394b74cf0f9476b2b81424f7a823dcd600bf66490445fa39090aba2824c795` |

Vision configuration:

- model: `qwen3-vl:8b-instruct`
- temperature: `0`
- context: `8192`
- structured JSON output
- visual role: literal transcription only
- no manual/oracle image used as production input

Results:

- frame count: `7`
- successful vision calls: `7`
- equation-bearing frames: `7`
- uncertain-token lists: empty for all 7 frames
- model-reported confidence: `1.0` for all 7 frames
- normalized ASCII groups: exactly one group with count `7`

Every frame returned the same ASCII transcription:

```text
B_0 = C/2 * [1 - (1 + YTM/2)^(-2t)] / (YTM/2) + F / (1 + YTM/2)^(2t)
```

Every frame returned the same LaTeX mathematical tree:

```tex
B_{0} = \frac{C}{2} \left[\frac{1 - \left(1+\frac{\text{YTM}}{2}\right)^{-2t}}{\frac{\text{YTM}}{2}}\right] + \frac{F}{\left(1+\frac{\text{YTM}}{2}\right)^{2t}}
```

The seven-frame agreement exceeds the provisional acceptance threshold of at least three independent equation-bearing frames preserving the same fraction nesting, additive terms, exponent signs, and denominator placement.

Important: model-reported confidence is recorded as provenance but is not itself an acceptance criterion. Production acceptance must be deterministic: parser-safe normalization, successful shared-AST parsing, and cross-frame structural agreement.

## Production acquisition and visual-consensus contract

For Stage E visual-equation recovery:

1. Resolve the visual cue window from transcript evidence.
2. Attempt `web_embedded` acquisition using yt-dlp native HTTP and a video-only format bounded to at most 1080p.
3. Persist source-media metadata and SHA-256 while the temporary media exists.
4. Extract bounded cue frames locally with ffmpeg using `-nostdin`.
5. Record frame timestamps and SHA-256 hashes.
6. Run literal visual transcription independently per frame with `qwen3-vl:8b-instruct`, temperature 0, concurrency 1.
7. Preserve the raw ASCII and LaTeX transcription for each frame immutably.
8. Normalize parser syntax into a separate field without silently modifying the raw visual evidence.
9. Parse normalized expressions with the shared safe AST parser.
10. Compare canonical AST structure across independently extracted frames.
11. Accept a visual formula only when the configured minimum number of independent frames agree structurally and all other deterministic validation gates pass.
12. Delete temporary full source media after accepted provenance is written.
13. If the video is not embeddable, acquisition fails, too few equation-bearing frames exist, parsing fails, or structural consensus is absent, fail closed to `visual_review_required`.

Do not silently fall back to the rejected `android_vr` or `mweb` paths.

## Provenance requirements for `visual_evidence.json`

The Stage E package should record at minimum:

- source URL and video ID
- calculation ID
- cue transcript segment IDs and start/end timestamps
- yt-dlp version
- selected YouTube client
- selected format ID, protocol, codec, dimensions, duration, and size
- acquisition mode/downloader
- source-media SHA-256
- frame timestamps and SHA-256 hashes
- vision model and model version/tag
- raw per-frame ASCII and LaTeX visual transcription
- uncertain tokens and model-reported confidence
- parser-safe normalized transcription as a distinct field
- parse result and canonical AST identity per frame
- cross-frame consensus group/count and required threshold
- final acceptance/review reason

## Next implementation stage

Stage E.1 and E.2 QC are accepted and frozen as behavioral requirements. The next implementation stage is to add the generalized production Stage E module and tests, integrate it with `CALC` entries marked `visual_equation_cue=true`, emit `visual_evidence.json`, and route accepted visual formulas into the existing shared AST/formula/coverage pipeline.

The v4.3 C-D.4C.1 source implementation remains the last frozen pre-Stage-E code checkpoint. This documentation checkpoint records accepted Stage E behavior but does not yet modify production research code.
