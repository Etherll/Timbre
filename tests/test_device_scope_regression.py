"""
Regression test for the DEVICE UnboundLocalError bug.

THE BUG (what this guards against):
    `DEVICE` is a module-level name in run_timbre.py (imported from common). The early
    device banner reads it: `log.info(f"... (Device: {DEVICE.type.upper()}) ...")`. main()
    ALSO reassigns DEVICE later (to force CPU under a CPU policy: `DEVICE = common.DEVICE`).
    In Python, ANY assignment to a name inside a function makes that name function-LOCAL for
    the WHOLE function body unless `global DEVICE` is declared. Without the declaration, the
    banner line -- which executes BEFORE the assignment -- reads a local that is not yet bound
    and raises `UnboundLocalError: local variable 'DEVICE' referenced before assignment`,
    crashing every single run before any work happens.

    The fix is the `global DEVICE` statement at the top of main(). This test asserts the fix
    is present using the `symtable` module on the SOURCE TEXT -- no import of run_timbre.py
    (which would drag in the heavy GPU/ML chain) and no GPU. symtable resolves Python scoping
    rules statically, so it sees exactly what the interpreter would: whether DEVICE binds to
    the module global or to a function local.

These tests are GPU-free and import-free by construction: they read run_timbre.py as text
and analyze it with `symtable`. They never import run_timbre / audio_pipeline.
"""
from __future__ import annotations

import symtable
from pathlib import Path

import pytest

# run_timbre.py lives at the repo root, one level up from tests/.
RUN_EXTRACTOR_PATH = Path(__file__).resolve().parent.parent / "run_timbre.py"


def _source() -> str:
    return RUN_EXTRACTOR_PATH.read_text(encoding="utf-8")


def _module_symtable() -> symtable.SymbolTable:
    return symtable.symtable(_source(), str(RUN_EXTRACTOR_PATH), "exec")


def _iter_function_symtables(st: symtable.SymbolTable):
    """Yield every function SymbolTable reachable from `st` (recursively, incl. nested defs)."""
    for child in st.get_children():
        if child.get_type() == "function":
            yield child
        yield from _iter_function_symtables(child)


def _find_function(st: symtable.SymbolTable, name: str):
    for fn in _iter_function_symtables(st):
        if fn.get_name() == name:
            return fn
    return None


def test_run_timbre_source_is_present():
    assert RUN_EXTRACTOR_PATH.is_file(), f"expected source at {RUN_EXTRACTOR_PATH}"


def test_main_declares_DEVICE_global_no_unbound_local():
    """
    The core regression: in main(), DEVICE must resolve to the module-global, NOT a
    function-local. is_global() True / is_local() False means main() rebinds the
    module-level DEVICE (via `global DEVICE`), so the early device banner that reads
    DEVICE cannot raise UnboundLocalError.
    """
    top = _module_symtable()
    main_st = _find_function(top, "main")
    assert main_st is not None, "could not locate main() symbol table in run_timbre.py"

    device = main_st.lookup("DEVICE")

    assert device.is_global() is True and device.is_local() is False, (
        "DEVICE UnboundLocalError REGRESSION: main() assigns DEVICE (to force CPU under a CPU "
        "policy) but does NOT declare `global DEVICE`, so DEVICE became function-LOCAL for the "
        "whole body. The early device banner reads DEVICE before that assignment and would "
        "raise `UnboundLocalError: local variable 'DEVICE' referenced before assignment`, "
        "crashing every run. Restore the `global DEVICE` statement at the top of main(). "
        f"(symtable saw is_global={device.is_global()}, is_local={device.is_local()})"
    )


def test_main_DEVICE_is_actually_read_and_assigned():
    """
    Guard the guard: this regression only matters because main() BOTH reads and assigns
    DEVICE. If a future refactor stops doing both, the test above is no longer meaningful;
    surface that here so the contract stays honest.
    """
    main_st = _find_function(_module_symtable(), "main")
    assert main_st is not None
    device = main_st.lookup("DEVICE")
    assert device.is_referenced() is True, (
        "Precondition gone: main() no longer READS DEVICE. The UnboundLocalError regression "
        "test assumes main() reads DEVICE (the device banner). Re-validate the regression."
    )
    assert device.is_assigned() is True, (
        "Precondition gone: main() no longer ASSIGNS DEVICE. The UnboundLocalError regression "
        "test assumes main() assigns DEVICE (CPU-policy override). Re-validate the regression."
    )


def test_every_function_that_reads_and_assigns_DEVICE_declares_it_global():
    """
    Defensive generalization that survives refactors: ANY function in run_timbre.py which
    BOTH reads and assigns the module-global DEVICE must declare `global DEVICE`. Otherwise
    that function reintroduces the exact UnboundLocalError bug class (read-before-assign of a
    name that became implicitly local). Checked across all functions (including nested ones)
    via symtable, so moving the assignment into a helper still gets caught.
    """
    top = _module_symtable()
    offenders = []
    for fn in _iter_function_symtables(top):
        try:
            sym = fn.lookup("DEVICE")
        except KeyError:
            continue  # function doesn't reference DEVICE at all
        reads = sym.is_referenced()
        assigns = sym.is_assigned()
        if reads and assigns and not sym.is_declared_global():
            offenders.append(fn.get_name())

    assert not offenders, (
        "UnboundLocalError bug class reintroduced: the following function(s) BOTH read and "
        f"assign the module-global DEVICE without declaring `global DEVICE`: {offenders}. "
        "Any read of DEVICE that executes before the assignment will raise "
        "`UnboundLocalError`. Add `global DEVICE` to each listed function (or stop assigning "
        "DEVICE inside it)."
    )


# Meta-test: prove the symtable check has teeth using synthetic source strings.
# (No production source is touched; these strings are analyzed in-memory.)
def _device_is_global_in_main(source: str) -> bool:
    """Return True iff DEVICE resolves to the module-global inside main() of `source`."""
    top = symtable.symtable(source, "<synthetic>", "exec")
    main_st = _find_function(top, "main")
    assert main_st is not None
    dev = main_st.lookup("DEVICE")
    return dev.is_global() and not dev.is_local()


_BUGGY_SOURCE = (
    "DEVICE = 'cuda'\n"
    "def main(args):\n"
    "    print(DEVICE)            # read BEFORE assignment -> UnboundLocalError at runtime\n"
    "    DEVICE = 'cpu'           # makes DEVICE function-local for the whole body\n"
)

_FIXED_SOURCE = (
    "DEVICE = 'cuda'\n"
    "def main(args):\n"
    "    global DEVICE            # the fix\n"
    "    print(DEVICE)            # now reads the module-global -> safe\n"
    "    DEVICE = 'cpu'\n"
)


def test_kill_query_checker_flags_buggy_source():
    """A synthetic main() that assigns DEVICE without `global` must be FLAGGED (returns False)."""
    assert _device_is_global_in_main(_BUGGY_SOURCE) is False, (
        "Checker is toothless: it failed to flag a main() that reassigns DEVICE without a "
        "`global` declaration (the exact bug)."
    )


def test_kill_query_checker_passes_correct_source():
    """A synthetic main() with `global DEVICE` must PASS (returns True)."""
    assert _device_is_global_in_main(_FIXED_SOURCE) is True, (
        "Checker is broken: it rejected a correct main() that declares `global DEVICE`."
    )
