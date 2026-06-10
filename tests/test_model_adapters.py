"""
Pure-logic tests for the new model-stack adapter helpers (no GPU / no heavy ML deps).

These cover the conversion seams introduced by the model-stack migration:
  * NeMo Sortformer segments -> pyannote Annotation (and overlap derived from it),
  * audio-separator vocals-stem selection,
  * FireRedVAD speech-ratio.
The actual model inference (Sortformer / audio-separator / FireRedVAD) needs a GPU box and
is verified manually; only pyannote.core (a light dep) is exercised here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from timbre import diarization as diar
from timbre import separation as sep
from timbre import vad


# diarization: NeMo segments -> Annotation, and overlap derivation
def test_normalize_segments_accepts_rttm_strings():
    segs = diar.normalize_segments(["0.00 5.00 speaker_0", "3.0 8.0 speaker_1"])
    assert segs == [(0.0, 5.0, "speaker_0"), (3.0, 8.0, "speaker_1")]


def test_normalize_segments_accepts_tuples_and_unwraps_per_file_list():
    # NeMo sometimes nests results one level per input file.
    nested = [[(0.0, 1.0, "s0"), (1.0, 2.0, "s1")]]
    assert diar.normalize_segments(nested) == [(0.0, 1.0, "s0"), (1.0, 2.0, "s1")]


def test_normalize_segments_skips_malformed_strings():
    assert diar.normalize_segments(["bad", "0 1 s0"]) == [(0.0, 1.0, "s0")]


def test_normalize_segments_empty():
    assert diar.normalize_segments(None) == []
    assert diar.normalize_segments([]) == []


def test_nemo_segments_to_annotation_labels_and_timeline():
    ann = diar.nemo_segments_to_annotation(
        ["0.00 5.00 speaker_0", "3.00 8.00 speaker_1", (10.0, 11.0, "speaker_0")]
    )
    assert set(ann.labels()) == {"speaker_0", "speaker_1"}
    spk0 = [(s.start, s.end) for s in ann.label_timeline("speaker_0")]
    assert (0.0, 5.0) in spk0 and (10.0, 11.0) in spk0


def test_nemo_segments_to_annotation_drops_nonpositive_spans():
    ann = diar.nemo_segments_to_annotation(["5.0 5.0 s0", "1.0 2.0 s1"])
    assert ann.labels() == ["s1"]


def test_overlap_from_diarization_matches_get_overlap():
    ann = diar.nemo_segments_to_annotation(
        ["0.0 5.0 speaker_0", "3.0 8.0 speaker_1", "10.0 11.0 speaker_0"]
    )
    overlap = diar.overlap_from_diarization(ann)
    regions = [(s.start, s.end) for s in overlap]
    assert regions == [(3.0, 5.0)]


def test_overlap_from_diarization_none_is_empty():
    assert len(diar.overlap_from_diarization(None)) == 0


def test_overlap_empty_when_no_simultaneous_speakers():
    ann = diar.nemo_segments_to_annotation(["0.0 2.0 s0", "3.0 4.0 s1"])
    assert len(diar.overlap_from_diarization(ann)) == 0


# separation: vocals-stem selection
def test_select_vocals_stem_prefers_vocals_tag():
    files = [
        "song_(Instrumental)_UVR-MDX-NET-Inst_HQ_3.wav",
        "song_(Vocals)_UVR-MDX-NET-Inst_HQ_3.wav",
    ]
    chosen = sep.select_vocals_stem(files)
    assert chosen is not None and "(Vocals)" in chosen.name


def test_select_vocals_stem_resolves_against_output_dir():
    chosen = sep.select_vocals_stem(["clip_(Vocals)_model.wav"], output_dir="/tmp/out")
    assert chosen == Path("/tmp/out") / "clip_(Vocals)_model.wav"


def test_select_vocals_stem_falls_back_to_vocal_substring():
    chosen = sep.select_vocals_stem(["a_vocals.wav", "a_other.wav"])
    assert chosen is not None and "vocals" in chosen.name.lower()


def test_select_vocals_stem_none_when_absent():
    assert sep.select_vocals_stem(["a_(Instrumental)_m.wav"]) is None
    assert sep.select_vocals_stem([]) is None


# vad: speech ratio
@pytest.mark.parametrize(
    "timestamps, total, expected",
    [
        ([(0.0, 1.0), (2.0, 3.0)], 4.0, 0.5),
        ([], 4.0, 0.0),
        ([(0.0, 2.0)], 0.0, 0.0),          # non-positive duration -> 0
        ([(0.0, 10.0)], 4.0, 1.0),         # clamped to 1.0
        ([(0.0, 0.0)], 4.0, 0.0),          # zero-length span
    ],
)
def test_speech_ratio(timestamps, total, expected):
    assert vad.speech_ratio(timestamps, total) == pytest.approx(expected)


def test_set_and_get_default_vad_model_dir():
    original = vad.get_default_model_dir()
    try:
        vad.set_default_model_dir("/custom/vad/dir")
        assert vad.get_default_model_dir() == "/custom/vad/dir"
        # falsy values do not override
        vad.set_default_model_dir("")
        assert vad.get_default_model_dir() == "/custom/vad/dir"
    finally:
        vad.set_default_model_dir(original)
