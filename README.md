# soccer-analysis

AI-powered soccer match analysis for amateur, youth, and high school teams. Coaches upload match
video (mainly Veo follow-cam exports) and get back stats, a 2D radar of player positions, and,
eventually, per-player coaching tips linked to video clips.

**Status:** early development (Phase 0: setup and baseline pipeline). See [docs/devlog.md](docs/devlog.md).

## License

Licensed under the [Apache License 2.0](LICENSE).

The project's own code and production pipeline use only permissively licensed dependencies (MIT, BSD,
Apache-2.0). The Ultralytics YOLO baseline used for early experiments is AGPL-3.0, so it is kept
outside the shipped code as a learning tool only. See [docs/licensing.md](docs/licensing.md).
