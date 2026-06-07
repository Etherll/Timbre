"""
AST structural contract on run_timbre.py main().

INVARIANT UNDER TEST: the DEFAULT (big-GPU, no --low-vram, fp32) path stays byte-for-byte
identical to the pre-policy behavior. The policy seam is allowed to ADD behavior under
non-default policies, but it must never change what happens on the default path. Two lexical
gates encode that invariant in the source:

  * `if not policy.defer_verification_models:`  -- the EAGER STAGE-0 verification-model init
    (WeSpeaker + SpeechBrain) only runs when NOT deferring. On the default path deferral is
    off, so the eager init runs exactly as before.

  * `if policy.free_between_stages:`             -- every inter-stage model free (loader
    `.unload()` + `runtime.free_model()`) only runs when freeing is enabled. On the default
    path freeing is off, so NO frees happen -- models stay resident, byte-for-byte as before.

If anyone removes a gate (e.g. makes the frees unconditional), the DEFAULT path changes and
this contract fails. The checks are AST-structural: they match on attribute/name identifiers
and enclosing `if`-tests, never on line numbers, so they survive reordering/reindenting.

GPU-free and import-free: run_timbre.py is parsed as TEXT with `ast`. We never import
run_timbre / audio_pipeline (the heavy ML chain).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUN_EXTRACTOR_PATH = Path(__file__).resolve().parent.parent / "run_timbre.py"


# AST helpers -- identifier-based, line-number-free.
def _parse_main() -> ast.FunctionDef:
    src = RUN_EXTRACTOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("could not find main() in run_timbre.py")


def _parents(root: ast.AST) -> dict:
    """child -> parent map for every node under `root`."""
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _call_attr(call: ast.Call) -> str | None:
    """Trailing identifier of a call: `runtime.free_model()` -> 'free_model', `f()` -> 'f'."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _call_dotted(call: ast.Call) -> str | None:
    """Dotted call spelling: `runtime.free_model()` -> 'runtime.free_model'; `f()` -> 'f'."""
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    if isinstance(f, ast.Attribute):
        return f".{f.attr}"
    if isinstance(f, ast.Name):
        return f.id
    return None


def _enclosing_if_references(node: ast.AST, token: str, parents: dict) -> bool:
    """
    True iff `node` is lexically nested under some enclosing `ast.If` whose *test* references
    `token` as an attribute (`policy.<token>`) or a bare name (`<token>`). Walks the full
    parent chain, so the gate may be several blocks up.
    """
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.If):
            test_names = {n.attr for n in ast.walk(cur.test) if isinstance(n, ast.Attribute)}
            test_names |= {n.id for n in ast.walk(cur.test) if isinstance(n, ast.Name)}
            if token in test_names:
                return True
        cur = parents.get(cur)
    return False


def _calls_named(scope: ast.AST, *attr_names: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(scope)
        if isinstance(n, ast.Call) and _call_attr(n) in attr_names
    ]


# Contract checks against the real source.
def test_run_timbre_source_is_present():
    assert RUN_EXTRACTOR_PATH.is_file(), f"expected source at {RUN_EXTRACTOR_PATH}"


def test_eager_stage0_verification_init_is_gated_by_defer_flag():
    """
    The eager STAGE-0 init calls -- get_wespeaker() and get_speechbrain() -- must each have at
    least one invocation nested under an `if` whose test references `defer_verification_models`
    (the `if not policy.defer_verification_models:` block). That is the eager-init site; the
    other (lazy) invocations just-before STAGE 5/6 are intentionally ungated. Removing the
    gate would make the eager init unconditional and change the deferral semantics.
    """
    main_fn = _parse_main()
    parents = _parents(main_fn)

    for loader in ("get_wespeaker", "get_speechbrain"):
        calls = _calls_named(main_fn, loader)
        assert calls, f"expected at least one call to {loader}() in main()"
        gated = [c for c in calls if _enclosing_if_references(c, "defer_verification_models", parents)]
        assert gated, (
            f"DEFAULT-path invariant broken: no call to {loader}() is guarded by an `if` "
            "referencing `defer_verification_models`. The eager STAGE-0 verification-model "
            "init must stay under `if not policy.defer_verification_models:` so it only runs "
            "when NOT deferring (i.e. unchanged on the default path)."
        )


def test_every_unload_and_free_model_call_is_gated_by_free_between_stages():
    """
    Every loader `.unload()` (transcription / diarization / vad) AND every `free_model(` /
    `runtime.free_model(` call inside main() must be lexically nested under an `if` whose test
    references `free_between_stages`. On the DEFAULT path free_between_stages is False, so NO
    frees run and models stay resident -- byte-for-byte as before. Making any free
    unconditional changes the default path and must fail here.
    """
    main_fn = _parse_main()
    parents = _parents(main_fn)

    free_calls = _calls_named(main_fn, "unload", "free_model")
    assert free_calls, "expected at least one unload()/free_model() call in main()"

    ungated = []
    for c in free_calls:
        if not _enclosing_if_references(c, "free_between_stages", parents):
            ungated.append((_call_dotted(c), getattr(c, "lineno", "?")))

    assert not ungated, (
        "DEFAULT-path invariant broken: the following model-free call(s) are NOT guarded by an "
        f"`if` referencing `free_between_stages`: {ungated}. Every `.unload()` and "
        "`free_model()` in main() must stay under `if policy.free_between_stages:` so the "
        "default path performs no inter-stage frees (models stay resident, behavior unchanged)."
    )


def test_exactly_one_resolve_policy_and_one_set_active_policy():
    """
    The per-run policy is resolved once and installed once. Exactly one `runtime.resolve_policy(`
    and exactly one `runtime.set_active_policy(` call in main(). Duplicates would mean the
    policy seam is being re-entered (a correctness smell); zero would mean the seam is gone.
    """
    main_fn = _parse_main()
    resolve = _calls_named(main_fn, "resolve_policy")
    set_active = _calls_named(main_fn, "set_active_policy")
    assert len(resolve) == 1, f"expected exactly one resolve_policy() call, found {len(resolve)}"
    assert len(set_active) == 1, f"expected exactly one set_active_policy() call, found {len(set_active)}"


# Kill-queries: prove the AST checkers have teeth. We feed synthetic main()
# bodies (no production source touched) and assert the checker FLAGS the
# gate-removed version and PASSES the correctly-gated version.
def _check_frees_all_gated(source: str) -> bool:
    """
    Standalone re-implementation of the free-gating check against an arbitrary source string.
    Returns True iff every unload()/free_model() call in main() is gated by `free_between_stages`.
    """
    tree = ast.parse(source)
    main_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    parents = _parents(main_fn)
    free_calls = _calls_named(main_fn, "unload", "free_model")
    return bool(free_calls) and all(
        _enclosing_if_references(c, "free_between_stages", parents) for c in free_calls
    )


def _check_eager_init_gated(source: str) -> bool:
    """
    True iff each of get_wespeaker/get_speechbrain has at least one call gated by
    `defer_verification_models` in main() of `source`.
    """
    tree = ast.parse(source)
    main_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    parents = _parents(main_fn)
    for loader in ("get_wespeaker", "get_speechbrain"):
        calls = _calls_named(main_fn, loader)
        if not calls:
            return False
        if not any(_enclosing_if_references(c, "defer_verification_models", parents) for c in calls):
            return False
    return True


_GATE_REMOVED_FREES = '''
def main(args):
    do_work()
    # GATE REMOVED: free is now unconditional -- changes the DEFAULT path.
    from timbre import diarization as _diar
    _diar.unload()
    runtime.free_model()
'''

_CORRECT_FREES = '''
def main(args):
    do_work()
    if policy.free_between_stages:
        from timbre import diarization as _diar
        _diar.unload()
        runtime.free_model()
'''

_GATE_REMOVED_EAGER = '''
def main(args):
    # GATE REMOVED: eager init is unconditional -- deferral semantics broken.
    wespeaker_models = get_wespeaker()
    speechbrain_model = get_speechbrain()
'''

_CORRECT_EAGER = '''
def main(args):
    if not policy.defer_verification_models:
        wespeaker_models = get_wespeaker()
        speechbrain_model = get_speechbrain()
    else:
        wespeaker_models = None
        speechbrain_model = None
'''


def test_kill_query_frees_gate_removed_is_flagged():
    """Removing the free_between_stages gate must make the checker return False (caught)."""
    assert _check_frees_all_gated(_GATE_REMOVED_FREES) is False, (
        "Checker is toothless: an unconditional unload()/free_model() (gate removed) was NOT "
        "flagged -- it would silently let the DEFAULT path change."
    )


def test_kill_query_frees_correct_passes():
    """Correctly gated frees must make the checker return True."""
    assert _check_frees_all_gated(_CORRECT_FREES) is True, (
        "Checker is broken: it rejected correctly free_between_stages-gated frees."
    )


def test_kill_query_eager_init_gate_removed_is_flagged():
    """Removing the defer_verification_models gate must make the eager-init checker return False."""
    assert _check_eager_init_gated(_GATE_REMOVED_EAGER) is False, (
        "Checker is toothless: an unconditional eager STAGE-0 init (gate removed) was NOT "
        "flagged."
    )


def test_kill_query_eager_init_correct_passes():
    """Correctly gated eager init must make the checker return True."""
    assert _check_eager_init_gated(_CORRECT_EAGER) is True, (
        "Checker is broken: it rejected a correctly defer_verification_models-gated eager init."
    )
