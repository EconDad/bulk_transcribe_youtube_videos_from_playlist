# Stage E Acquisition QC

Date: 2026-08-15
Branch baseline before this note: `5464ed5` (`research-v4.3-stage-cd4c1`)

## Status

Stage E.1 autonomous source-media acquisition is accepted for embeddable YouTube videos using the `web_embedded` client and yt-dlp's native HTTP downloader. No Stage E production source implementation is included in this checkpoint; this file records the accepted acquisition contract and QC evidence only.

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

## Accepted path: `web_embedded` + native HTTP

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

Accepted frame evidence:

- timestamp: `785.5` seconds
- frame SHA-256: `5261173c9e8cbb80d0c7557acf851a8fd611cb1c611c660d410dd76d8ca3c146`

## Production acquisition contract

For Stage E visual-equation recovery:

1. Resolve the visual cue window from transcript evidence.
2. Attempt `web_embedded` acquisition using yt-dlp native HTTP and a video-only format bounded to at most 1080p.
3. Persist source-media metadata and SHA-256 while the temporary media exists.
4. Extract only the bounded cue frames locally with ffmpeg.
5. Record frame timestamps and SHA-256 hashes.
6. Run visual transcription independently per frame.
7. Normalize visual transcription separately from raw evidence, then validate through the shared safe AST parser and cross-frame structural agreement.
8. Delete temporary full source media after accepted provenance is written.
9. If the video is not embeddable or acquisition otherwise fails, fail closed to `visual_review_required` until another acquisition backend is explicitly accepted.

Do not silently fall back to the rejected `android_vr` or `mweb` paths.

## Provenance requirements for `visual_evidence.json`

The Stage E package should record at minimum:

- source URL and video ID
- cue transcript segment IDs and start/end timestamps
- yt-dlp version
- selected YouTube client
- selected format ID, protocol, codec, dimensions, duration, and size
- acquisition mode/downloader
- source-media SHA-256
- frame timestamps and SHA-256 hashes
- vision model and model version/tag
- raw per-frame visual transcription
- parser-safe normalized transcription as a distinct field
- cross-frame consensus result
- final acceptance/review reason

## Stage E.2 state

A first autonomous multi-frame extraction command was started after acceptance of `web_embedded`, but the pasted shell block became corrupted during execution. Seven local frames were extracted, but the command terminated before the vision-consensus phase completed. Therefore Stage E.2 is **not accepted** by this checkpoint and those partial outputs are not treated as authoritative evidence.

The frozen source checkpoint remains the v4.3 C-D.4C.1 implementation; this documentation commit records only the accepted Stage E.1 acquisition behavior and does not modify production research code.
