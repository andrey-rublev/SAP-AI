"""One desktop-input owner per Windows session, across SAP-AI checkouts.

Importing this module performs no native calls. Hold ``desktop_control_lease``
through controller startup, its captures and input, including launch/restarts.
Passive preview, snapshot capture and offline inspection need no input lease.
The kernel releases the named mutex and its handles if its owning process
crashes; there are no PID files, stale-lock deletion or process termination.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os


DESKTOP_INPUT_MUTEX_NAME = r"Local\SAPAI.DesktopInput.v1"
ERROR_ALREADY_EXISTS = 183


class DesktopControlError(RuntimeError):
    """The desktop-input lease could not be acquired or safely cleaned up."""


class DesktopControlConflict(DesktopControlError):
    """Another controller, including a nested caller, already holds the lease."""


def _load_kernel32():
    if os.name != "nt":
        raise DesktopControlError("the desktop-input lease is available only on Windows")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _signatures(kernel32):
    # HANDLE must remain pointer-sized on 64-bit Windows; an implicit ctypes
    # int return would truncate a valid mutex handle.
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    for name in ("ReleaseMutex", "CloseHandle"):
        function = getattr(kernel32, name)
        function.argtypes = [wintypes.HANDLE]
        function.restype = wintypes.BOOL
    kernel32.SetLastError.argtypes = [wintypes.DWORD]
    kernel32.SetLastError.restype = None
    kernel32.GetLastError.argtypes = []
    kernel32.GetLastError.restype = wintypes.DWORD


def _note(exception, message):
    # Preserve the original body/conflict exception if cleanup also fails.
    # add_note is available on current Python and optional on Python 3.10.
    if hasattr(exception, "add_note"):
        exception.add_note(message)


@contextmanager
def desktop_control_lease(*, kernel32=None):
    """Own the fixed session-wide input lease for the context's lifetime.

    ``kernel32`` is an injectable Win32 API for tests; normal callers omit it.
    Acquisition never waits. CreateMutexW grants initial ownership only when it
    creates the named object. ERROR_ALREADY_EXISTS is rejected even on the same
    thread, avoiding Win32 mutexes' otherwise recursive ownership semantics.
    Enter and exit the context on the same thread, as Win32 ownership requires.
    """
    kernel32 = _load_kernel32() if kernel32 is None else kernel32
    _signatures(kernel32)
    kernel32.SetLastError(0)
    handle = kernel32.CreateMutexW(None, True, DESKTOP_INPUT_MUTEX_NAME)
    error = kernel32.GetLastError()
    if not handle:
        raise DesktopControlError(f"cannot acquire the Windows desktop-input lease (Win32 error {error})")
    if error == ERROR_ALREADY_EXISTS:
        conflict = DesktopControlConflict(
            "another SAP-AI controller already holds the desktop-input lease in this Windows session; "
            "stop that controller before retrying (nested leases are also refused)"
        )
        if not kernel32.CloseHandle(handle):
            _note(conflict, f"could not close the rejected mutex handle (Win32 error {kernel32.GetLastError()})")
        raise conflict
    body_error = None
    try:
        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        failures = []
        try:
            if not kernel32.ReleaseMutex(handle):
                failures.append(f"ReleaseMutex failed (Win32 error {kernel32.GetLastError()})")
        finally:
            if not kernel32.CloseHandle(handle):
                failures.append(f"CloseHandle failed (Win32 error {kernel32.GetLastError()})")
        if failures:
            message = "desktop-input lease cleanup: " + "; ".join(failures)
            if body_error is not None:
                _note(body_error, message)
            else:
                raise DesktopControlError(message)
