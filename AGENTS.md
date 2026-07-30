# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
Single Python 3.12 CLI (`bulk_transcribe_youtube_videos_from_playlist.py`) that downloads YouTube audio, transcribes it with `faster-whisper` (`large-v3`), splits sentences with spaCy, and writes `.txt`/`.csv`/`.json`. Plus one static page, `transcript_reader.html`, that cleans/formats a transcript in the browser. No monorepo, no database, no backend server, no test suite.

### Environment layout
- Dependencies live in a venv at `./venv` (git-ignored). The startup update script maintains it. Run everything via `./venv/bin/python` / `./venv/bin/ruff`, or `source venv/bin/activate` first.
- `python3.12-venv` (system apt package) is required to create the venv and is already present in the VM snapshot; it is intentionally NOT in the update script.
- The spaCy model `en_core_web_sm` is installed by the update script (the app auto-downloads it on first use otherwise).

### Lint / run
- Lint: `./venv/bin/ruff check .` — the repo currently reports ~11 pre-existing style findings (do not "fix" as part of unrelated work). `E501` is ignored via `pyproject.toml`.
- Run the app: edit the config constants at the top of `bulk_transcribe_youtube_videos_from_playlist.py` (e.g. `single_video_url`, `convert_single_video`), then `./venv/bin/python bulk_transcribe_youtube_videos_from_playlist.py`. Outputs land in `downloaded_audio/`, `generated_transcript_combined_texts/`, `generated_transcript_metadata_tables/` (all git-ignored).
- Reader page: `./venv/bin/python -m http.server 8000` then open `http://localhost:8000/transcript_reader.html`. It pulls Bootstrap/jQuery/Compromise from CDNs (needs egress).

### Load-bearing gotchas
- YouTube downloads via `pytubefix` FAIL from this cloud VM with `pytubefix.exceptions.BotDetection`, even with a valid PO token. The bundled node BotGuard PO-token generator works (returns a ~116-char token) and `visitor_data` resolves, but YouTube still rejects the request — this is IP-level blocking of the datacenter IP, not a missing dependency or token, and is not fixable via environment setup. General internet egress otherwise works.
- To exercise the transcription pipeline end-to-end without YouTube, feed a local audio file to `compute_transcript_with_whisper_from_audio_func(path, name, size_mb)` (async). This is how the environment was validated (an 11s local WAV transcribed correctly).
- First local transcription downloads `faster-whisper` `large-v3` weights (multi-GB) to the HF cache. No GPU is present, so it runs on CPU (`device=cpu`, `compute_type=auto`); expect CPU-speed inference.
- `.gitignore` ignores `*.md`, so committing `AGENTS.md`/`readme.md` requires `git add -f`.
