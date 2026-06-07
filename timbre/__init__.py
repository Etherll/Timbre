"""
timbre — identify, isolate, and transcribe clean solo segments of a target
speaker from multi-speaker audio.

This package is the refactored, modular home of what previously lived in three
monolithic modules (common.py, audio_pipeline.py, run_timbre.py). It is organized
so that new pipeline stages and model backends can be added in isolation:

    timbre/
        constants.py      frozen defaults / weights / thresholds (single source of truth)
        config.py         ExtractorConfig dataclass built from the CLI
        cli.py            argparse surface + main() entrypoint
        naming.py         filename / duration formatting (pure)
        segments.py       segment timeline logic (pure, no GPU/ML deps)
        verification.py   speaker-verification score fusion (pure)
        audio/            audio math + I/O helpers
        runtime/          (planned) lazy dependency + device + logging accessors
        bootstrap/        (planned) explicit environment bootstrap (packages/repos/models)
        models/           model-adapter protocols + a name->backend registry
        pipeline/         Stage protocol, PipelineContext, Orchestrator
        stages/           one Stage wrapper per pipeline step

Importing this package has NO heavy side effects: nothing here imports torch, whisper,
or pyannote.audio at module top, and no dependency bootstrapping runs on import.
"""

__version__ = "0.2.0"
