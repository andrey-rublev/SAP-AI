"""Named-mutex protocol tests with fake kernel32; no native locks or input."""
from ctypes import wintypes
import threading
from unittest.mock import Mock

import pytest

import sap_ai.desktop.control as control
class FakeFunction:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.function(*args)


class MutexRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_handle = 0x1_0000_0100
        self.names = {}
        self.handles = {}


class FakeKernel32:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else MutexRegistry()
        self.local = threading.local()
        self.CreateMutexW = FakeFunction(self.create)
        self.ReleaseMutex = FakeFunction(self.release)
        self.CloseHandle = FakeFunction(self.close)
        self.GetLastError = FakeFunction(lambda: getattr(self.local, "error", 0))
        self.SetLastError = FakeFunction(lambda value: setattr(self.local, "error", value))

    def create(self, security, initial_owner, name):
        with self.registry.lock:
            existing = self.registry.names.get(name)
            if existing is None:
                existing = {"name": name, "owner": threading.get_ident() if initial_owner else None, "handles": 0}
                self.registry.names[name] = existing
                self.local.error = 0
            else:
                self.local.error = control.ERROR_ALREADY_EXISTS
            handle = self.registry.next_handle
            self.registry.next_handle += 1
            self.registry.handles[handle] = existing
            existing["handles"] += 1
            return handle

    def release(self, handle):
        with self.registry.lock:
            item = self.registry.handles[handle]
            if item["owner"] != threading.get_ident():
                self.local.error = 288  # ERROR_NOT_OWNER
                return False
            item["owner"] = None
            return True

    def close(self, handle):
        with self.registry.lock:
            item = self.registry.handles.pop(handle)
            item["handles"] -= 1
            if item["handles"] == 0:
                del self.registry.names[item["name"]]
            return True


def test_lease_holds_fixed_named_mutex_and_preserves_64_bit_handle():
    kernel = FakeKernel32()
    with control.desktop_control_lease(kernel32=kernel):
        assert kernel.CreateMutexW.calls == [(None, True, r"Local\SAPAI.DesktopInput.v1")]
        assert kernel.ReleaseMutex.calls == kernel.CloseHandle.calls == []
        handle = next(iter(kernel.registry.handles))
        assert handle > 2**32
    assert kernel.ReleaseMutex.calls == kernel.CloseHandle.calls == [(handle,)]
    assert not kernel.registry.handles and not kernel.registry.names
    assert kernel.CreateMutexW.restype is wintypes.HANDLE
    assert kernel.CreateMutexW.argtypes == [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    assert kernel.ReleaseMutex.argtypes == kernel.CloseHandle.argtypes == [wintypes.HANDLE]
    assert kernel.ReleaseMutex.restype is kernel.CloseHandle.restype is wintypes.BOOL


def test_nested_same_thread_context_is_refused_without_releasing_its_owner():
    kernel = FakeKernel32()
    with control.desktop_control_lease(kernel32=kernel):
        owner_handle = next(iter(kernel.registry.handles))
        with pytest.raises(control.DesktopControlConflict, match="nested leases"):
            with control.desktop_control_lease(kernel32=kernel):
                pytest.fail("nested caller entered")
        assert kernel.ReleaseMutex.calls == []
        assert set(kernel.registry.handles) == {owner_handle}
        assert len(kernel.CloseHandle.calls) == 1
    assert len(kernel.ReleaseMutex.calls) == 1 and len(kernel.CloseHandle.calls) == 2


def test_independent_controllers_share_the_name_and_conflict_without_waiting():
    registry = MutexRegistry()
    first, second = FakeKernel32(registry), FakeKernel32(registry)
    with control.desktop_control_lease(kernel32=first):
        with pytest.raises(control.DesktopControlConflict, match="stop that controller"):
            with control.desktop_control_lease(kernel32=second):
                pytest.fail("second controller entered")
        assert second.ReleaseMutex.calls == [] and len(second.CloseHandle.calls) == 1
    with control.desktop_control_lease(kernel32=second):
        assert len(registry.names) == 1
    assert not registry.handles


def test_other_thread_is_refused_and_does_not_disturb_owner():
    kernel = FakeKernel32()
    errors = []
    def contender():
        try:
            with control.desktop_control_lease(kernel32=kernel):
                errors.append("unexpected entry")
        except control.DesktopControlConflict:
            errors.append("conflict")
    with control.desktop_control_lease(kernel32=kernel):
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == ["conflict"] and kernel.ReleaseMutex.calls == []
    assert not kernel.registry.handles


@pytest.mark.parametrize("failure", [RuntimeError("controller failed"), KeyboardInterrupt(), SystemExit(3)])
def test_body_exceptions_release_and_close_before_propagating(failure):
    kernel = FakeKernel32()
    with pytest.raises(type(failure)) as caught:
        with control.desktop_control_lease(kernel32=kernel):
            raise failure
    assert caught.value is failure
    assert len(kernel.ReleaseMutex.calls) == len(kernel.CloseHandle.calls) == 1
    assert not kernel.registry.handles
    # A normal subsequent controller can own the same name without stale cleanup.
    with control.desktop_control_lease(kernel32=kernel):
        pass


def test_failed_creation_reports_win32_error_and_never_releases_invalid_handle():
    kernel = FakeKernel32()
    def denied(*_):
        kernel.local.error = 5
        return 0
    kernel.CreateMutexW.function = denied
    with pytest.raises(control.DesktopControlError, match="acquire.*Win32 error 5"):
        with control.desktop_control_lease(kernel32=kernel):
            pytest.fail("failed creation entered")
    assert kernel.ReleaseMutex.calls == kernel.CloseHandle.calls == []


def test_old_last_error_does_not_turn_a_new_mutex_into_a_conflict():
    kernel = FakeKernel32()
    kernel.local.error = control.ERROR_ALREADY_EXISTS
    create = kernel.CreateMutexW.function
    def stale_success(*args):
        prior = kernel.local.error
        handle = create(*args)
        kernel.local.error = prior
        return handle
    kernel.CreateMutexW.function = stale_success
    with control.desktop_control_lease(kernel32=kernel):
        assert kernel.SetLastError.calls == [(0,)]


def test_release_failure_still_closes_and_reports_cleanup_error():
    kernel = FakeKernel32()
    def failed_release(_):
        kernel.local.error = 288
        return False
    kernel.ReleaseMutex.function = failed_release
    with pytest.raises(control.DesktopControlError, match="ReleaseMutex failed.*288"):
        with control.desktop_control_lease(kernel32=kernel):
            pass
    assert len(kernel.CloseHandle.calls) == 1


def test_cleanup_failures_preserve_original_controller_exception():
    kernel = FakeKernel32()
    kernel.ReleaseMutex.function = lambda _: False
    original_close = kernel.CloseHandle.function
    def failed_close(handle):
        original_close(handle)
        kernel.local.error = 6
        return False
    kernel.CloseHandle.function = failed_close
    failure = ValueError("original failure")
    with pytest.raises(ValueError, match="original failure") as caught:
        with control.desktop_control_lease(kernel32=kernel):
            raise failure
    assert caught.value is failure
    assert len(kernel.ReleaseMutex.calls) == len(kernel.CloseHandle.calls) == 1
    if hasattr(failure, "__notes__"):
        assert "CloseHandle failed (Win32 error 6)" in failure.__notes__[0]


def test_default_entry_loads_kernel32_lazily_and_can_be_replaced_without_native_calls(monkeypatch):
    kernel = FakeKernel32()
    loader = Mock(return_value=kernel)
    monkeypatch.setattr(control, "_load_kernel32", loader)
    context = control.desktop_control_lease()
    loader.assert_not_called()
    with context:
        loader.assert_called_once_with()


def test_real_loader_refuses_non_windows_before_loading_native_dll(monkeypatch):
    loader = Mock(side_effect=AssertionError("native DLL must not load"))
    monkeypatch.setattr(control.os, "name", "posix")
    monkeypatch.setattr(control.ctypes, "WinDLL", loader, raising=False)
    with pytest.raises(control.DesktopControlError, match="only on Windows"):
        with control.desktop_control_lease():
            pytest.fail("non-Windows entry")
    loader.assert_not_called()
