from __future__ import annotations

"""FieldLab-specific bounded semantic execution carrier inside Hermes Gateway.

This module is intentionally sibling-isolated from Supply.  It owns no FieldLab
business state, calculation, workflow, retry ledger, or credentials.  The only
business payload accepted is the exact existing FieldLab SemanticCapsuleV2.

Transport shape:
    {
      "operation_id": "fieldlab.semantic.execute.v1",
      "execution_id": "...",
      "timeout": <seconds>,
      "capsule": <exact SemanticCapsuleV2 object>
    }

The socket is a dedicated local IPC surface.  Its parent directory and ACL are
deployment concerns; this module never creates/chowns/chmods that directory and
never grants itself access to FieldLab paths or Supply paths.
"""

import json
import logging
import os
import re
import socketserver
import stat
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger("hermes.gateway.fieldlab_semantic_carrier")

FIELDLAB_OPERATION_ID = "fieldlab.semantic.execute.v1"
DEFAULT_FIELDLAB_SOCKET_PATH = Path("/run/hermes-fieldlab/semantic.sock")
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_TIMEOUT_SECONDS = 45.0
_EXECUTION_ID_RE = re.compile(r"[A-Za-z0-9._:-]{8,128}")
_ALLOWED_STAGES = frozenset({"PERSON", "ANDU_THEORY", "PATROL"})

FIELDLAB_SEMANTIC_SYSTEM_PROMPT = """You are the bounded semantic execution carrier for FieldLab.
You receive exactly one FieldLab SemanticCapsuleV2 as the user message.
Use only the evidence, rule_contract, output_contract, forbidden_information,
runtime_identity, provenance and semantic_stage contained in that capsule.
Do not use tools, memory, session history, external data, or unstated facts.
Do not become a FieldLab business authority, calculation owner, state owner,
write owner, workflow owner, or retry owner.
Perform only the qualitative semantic judgment requested by the capsule.
Respect every forbidden-information boundary.  If the authorized evidence is
insufficient, preserve the existing unresolved behavior rather than inventing
facts.  Return exactly one JSON object containing only the stage semantic result
required by the capsule output_contract.  Return no transport envelope, prose,
markdown, code fences, commentary, or hidden metadata."""


class FieldLabSemanticCarrierError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _validate_capsule(capsule: Any) -> Mapping[str, Any]:
    if not isinstance(capsule, Mapping):
        raise FieldLabSemanticCarrierError("INVALID_SEMANTIC_CAPSULE")
    if capsule.get("capsule_version") != 2:
        raise FieldLabSemanticCarrierError("SEMANTIC_CAPSULE_VERSION_MISMATCH")
    if capsule.get("semantic_stage") not in _ALLOWED_STAGES:
        raise FieldLabSemanticCarrierError("SEMANTIC_STAGE_NOT_ALLOWED")
    if capsule.get("authority") is not False:
        raise FieldLabSemanticCarrierError("SEMANTIC_CAPSULE_AUTHORITY_INVALID")
    if not _is_hash(capsule.get("input_capsule_hash")):
        raise FieldLabSemanticCarrierError("SEMANTIC_CAPSULE_HASH_INVALID")
    if not isinstance(capsule.get("runtime_identity"), Mapping):
        raise FieldLabSemanticCarrierError("SEMANTIC_RUNTIME_IDENTITY_INVALID")
    if not isinstance(capsule.get("evidence"), Mapping):
        raise FieldLabSemanticCarrierError("SEMANTIC_EVIDENCE_INVALID")
    if not isinstance(capsule.get("rule_contract"), list) or not capsule.get("rule_contract"):
        raise FieldLabSemanticCarrierError("SEMANTIC_RULE_CONTRACT_INVALID")
    if not isinstance(capsule.get("output_contract"), Mapping):
        raise FieldLabSemanticCarrierError("SEMANTIC_OUTPUT_CONTRACT_INVALID")
    if not isinstance(capsule.get("forbidden_information"), list):
        raise FieldLabSemanticCarrierError("SEMANTIC_FORBIDDEN_INFORMATION_INVALID")
    if not isinstance(capsule.get("provenance"), Mapping):
        raise FieldLabSemanticCarrierError("SEMANTIC_PROVENANCE_INVALID")
    return capsule


def _validate_request(
    request: Any,
) -> tuple[str, str, float, Mapping[str, Any]]:
    if not isinstance(request, Mapping):
        raise FieldLabSemanticCarrierError("INVALID_REQUEST")
    if set(request) != {"operation_id", "execution_id", "timeout", "capsule"}:
        raise FieldLabSemanticCarrierError("INVALID_REQUEST_SHAPE")
    operation_id = request.get("operation_id")
    if operation_id != FIELDLAB_OPERATION_ID:
        raise FieldLabSemanticCarrierError("UNKNOWN_OPERATION")
    execution_id = request.get("execution_id")
    if not isinstance(execution_id, str) or not _EXECUTION_ID_RE.fullmatch(execution_id):
        raise FieldLabSemanticCarrierError("INVALID_EXECUTION_ID")
    timeout = request.get("timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise FieldLabSemanticCarrierError("INVALID_TIMEOUT")
    timeout_f = float(timeout)
    if timeout_f <= 0 or timeout_f > MAX_TIMEOUT_SECONDS:
        raise FieldLabSemanticCarrierError("INVALID_TIMEOUT")
    capsule = _validate_capsule(request.get("capsule"))
    return operation_id, execution_id, timeout_f, capsule


def _configured_model_and_runtime() -> tuple[str, dict[str, Any]]:
    """Resolve exactly the configured Hermes provider/model; never auto-fallback."""

    try:
        from hermes_cli.config import load_config, split_model_config_default
        from hermes_cli.runtime_provider import resolve_runtime_provider
    except ImportError as exc:
        raise FieldLabSemanticCarrierError("HERMES_RUNTIME_UNAVAILABLE") from exc

    cfg = load_config() or {}
    model_cfg = cfg.get("model") or {}
    configured_provider = ""
    if isinstance(model_cfg, str):
        model = model_cfg.strip()
    elif isinstance(model_cfg, Mapping):
        raw_model = model_cfg.get("default") or model_cfg.get("model") or ""
        if isinstance(raw_model, Mapping):
            model, _ = split_model_config_default(raw_model)
        else:
            model = str(raw_model or "").strip()
        configured_provider = str(model_cfg.get("provider") or "").strip().lower()
    else:
        model = ""

    if not model:
        raise FieldLabSemanticCarrierError("HERMES_MODEL_NOT_CONFIGURED")
    if not configured_provider or configured_provider == "auto":
        raise FieldLabSemanticCarrierError("FIELDLAB_PROVIDER_MUST_BE_EXPLICIT")
    if configured_provider == "moa":
        raise FieldLabSemanticCarrierError("FIELDLAB_MULTI_MODEL_PROVIDER_FORBIDDEN")

    try:
        runtime = resolve_runtime_provider(
            requested=configured_provider,
            target_model=model,
        )
    except Exception as exc:
        raise FieldLabSemanticCarrierError("HERMES_PROVIDER_RESOLUTION_FAILED") from exc
    if not isinstance(runtime, dict):
        raise FieldLabSemanticCarrierError("HERMES_PROVIDER_RUNTIME_INVALID")
    if str(runtime.get("provider") or "").strip().lower() in {"", "auto", "moa"}:
        raise FieldLabSemanticCarrierError("HERMES_PROVIDER_RUNTIME_INVALID")
    return model, runtime


def _harden_agent_runtime(agent: Any) -> None:
    """Operation-local hardening; never mutates Hermes global provider policy."""

    agent._api_max_retries = 1
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_index = 0
    agent._budget_grace_call = False
    agent.compression_enabled = False
    agent._intent_ack_continuation = False
    agent._tool_use_enforcement = False
    agent._execution_guidance = False
    agent._task_completion_guidance = False
    agent._parallel_tool_call_guidance = False
    agent._environment_probe = False
    agent._bot_mode_protocol = False
    agent._persist_disabled = True
    agent.ephemeral_system_prompt = None
    agent._cached_system_prompt = FIELDLAB_SEMANTIC_SYSTEM_PROMPT


def _assert_agent_isolation(agent: Any) -> None:
    if getattr(agent, "enabled_toolsets", None) != []:
        raise FieldLabSemanticCarrierError("FIELDLAB_TOOLS_NOT_DISABLED")
    if getattr(agent, "tools", None):
        raise FieldLabSemanticCarrierError("FIELDLAB_TOOLS_NOT_DISABLED")
    if getattr(agent, "_session_db", None) is not None:
        raise FieldLabSemanticCarrierError("FIELDLAB_SESSION_STATE_NOT_DISABLED")
    if bool(getattr(agent, "_memory_enabled", False)) or bool(
        getattr(agent, "_user_profile_enabled", False)
    ):
        raise FieldLabSemanticCarrierError("FIELDLAB_MEMORY_NOT_DISABLED")
    if getattr(agent, "_memory_manager", None) is not None:
        raise FieldLabSemanticCarrierError("FIELDLAB_MEMORY_NOT_DISABLED")
    if not bool(getattr(agent, "skip_context_files", False)):
        raise FieldLabSemanticCarrierError("FIELDLAB_CONTEXT_FILES_NOT_DISABLED")
    if not bool(getattr(agent, "skip_background_review", False)):
        raise FieldLabSemanticCarrierError("FIELDLAB_BACKGROUND_REVIEW_NOT_DISABLED")
    if getattr(agent, "max_iterations", None) != 1:
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_ATTEMPT_BUDGET_INVALID")
    budget = getattr(agent, "iteration_budget", None)
    if budget is None or getattr(budget, "max_total", None) != 1:
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_ATTEMPT_BUDGET_INVALID")
    if getattr(budget, "used", 0) != 0:
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_ATTEMPT_ALREADY_USED")
    if getattr(agent, "_api_max_retries", None) != 1:
        raise FieldLabSemanticCarrierError("FIELDLAB_PROVIDER_RETRY_BUDGET_INVALID")
    if getattr(agent, "_fallback_chain", None):
        raise FieldLabSemanticCarrierError("FIELDLAB_PROVIDER_FALLBACK_NOT_DISABLED")
    if getattr(agent, "_fallback_model", None) is not None:
        raise FieldLabSemanticCarrierError("FIELDLAB_PROVIDER_FALLBACK_NOT_DISABLED")
    if bool(getattr(agent, "compression_enabled", True)):
        raise FieldLabSemanticCarrierError("FIELDLAB_COMPRESSION_NOT_DISABLED")
    if bool(getattr(agent, "_budget_grace_call", False)):
        raise FieldLabSemanticCarrierError("FIELDLAB_GRACE_CALL_NOT_DISABLED")
    if not bool(getattr(agent, "_persist_disabled", False)):
        raise FieldLabSemanticCarrierError("FIELDLAB_PERSISTENCE_NOT_DISABLED")
    if getattr(agent, "_cached_system_prompt", None) != FIELDLAB_SEMANTIC_SYSTEM_PROMPT:
        raise FieldLabSemanticCarrierError("FIELDLAB_SYSTEM_PROMPT_NOT_ISOLATED")
    if getattr(agent, "ephemeral_system_prompt", None):
        raise FieldLabSemanticCarrierError("FIELDLAB_SYSTEM_PROMPT_NOT_ISOLATED")


def build_fieldlab_isolated_agent(timeout: float) -> Any:
    model, runtime = _configured_model_and_runtime()
    try:
        from run_agent import AIAgent
    except ImportError as exc:
        raise FieldLabSemanticCarrierError("HERMES_AGENT_UNAVAILABLE") from exc

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        credential_pool=runtime.get("credential_pool"),
        request_overrides=runtime.get("request_overrides") or {},
        enabled_toolsets=[],
        quiet_mode=True,
        tool_progress_mode="off",
        max_iterations=1,
        run_budget_seconds=float(timeout),
        platform="fieldlab-semantic-carrier",
        skip_context_files=True,
        load_soul_identity=False,
        skip_memory=True,
        skip_background_review=True,
        session_db=None,
        save_trajectories=False,
        checkpoints_enabled=False,
        pass_session_id=False,
        log_prefix_chars=0,
        fallback_model=None,
    )
    _harden_agent_runtime(agent)
    try:
        _assert_agent_isolation(agent)
    except Exception:
        try:
            agent.close()
        finally:
            raise
    return agent


def execute_semantic_capsule(
    capsule: Mapping[str, Any],
    *,
    timeout: float,
    agent_factory: Callable[[float], Any] = build_fieldlab_isolated_agent,
) -> dict[str, Any]:
    """Execute one capsule with exactly one model attempt and strict JSON output."""

    capsule = _validate_capsule(capsule)
    prompt = json.dumps(
        capsule,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    agent = agent_factory(float(timeout))
    _harden_agent_runtime(agent)
    _assert_agent_isolation(agent)
    try:
        response = agent.run_conversation(prompt)
    except Exception as exc:
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_EXECUTION_FAILED") from exc
    finally:
        try:
            agent.close()
        except Exception:
            pass

    if not isinstance(response, Mapping):
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_RESPONSE_INVALID")
    api_calls = response.get("api_calls")
    if isinstance(api_calls, bool) or not isinstance(api_calls, int) or api_calls != 1:
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_ATTEMPT_COUNT_INVALID")
    raw = response.get("final_response")
    if not isinstance(raw, str):
        raise FieldLabSemanticCarrierError("FIELDLAB_MODEL_RESPONSE_INVALID")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FieldLabSemanticCarrierError("FIELDLAB_STAGE_RESULT_NOT_JSON") from exc
    if not isinstance(parsed, dict):
        raise FieldLabSemanticCarrierError("FIELDLAB_STAGE_RESULT_NOT_OBJECT")
    return parsed


def dispatch_fieldlab_semantic_request(
    request: Mapping[str, Any],
    *,
    executor: Callable[..., dict[str, Any]] = execute_semantic_capsule,
) -> dict[str, Any]:
    operation_id = FIELDLAB_OPERATION_ID
    execution_id = ""
    started = time.monotonic()
    try:
        operation_id, execution_id, timeout, capsule = _validate_request(request)
        result = executor(capsule, timeout=timeout)
        duration_ms = max(0, int((time.monotonic() - started) * 1000))
        LOGGER.info(
            "FieldLab semantic execution complete operation=%s execution=%s stage=%s "
            "capsule_hash=%s duration_ms=%d",
            operation_id,
            execution_id,
            capsule.get("semantic_stage"),
            capsule.get("input_capsule_hash"),
            duration_ms,
        )
        return {
            "ok": True,
            "operation_id": operation_id,
            "execution_id": execution_id,
            "result": result,
        }
    except FieldLabSemanticCarrierError as exc:
        LOGGER.warning(
            "FieldLab semantic execution blocked operation=%s execution=%s error=%s",
            operation_id,
            execution_id,
            exc.code,
        )
        return {
            "ok": False,
            "operation_id": operation_id,
            "execution_id": execution_id,
            "error": exc.code,
        }
    except Exception:
        LOGGER.exception(
            "FieldLab semantic execution failed operation=%s execution=%s",
            operation_id,
            execution_id,
        )
        return {
            "ok": False,
            "operation_id": operation_id,
            "execution_id": execution_id,
            "error": "FIELDLAB_SEMANTIC_CARRIER_INTERNAL_ERROR",
        }


class _FieldLabSemanticRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if not raw or len(raw) > MAX_MESSAGE_BYTES:
            self._reply(
                {
                    "ok": False,
                    "operation_id": FIELDLAB_OPERATION_ID,
                    "execution_id": "",
                    "error": "FIELDLAB_REQUEST_SIZE_INVALID",
                }
            )
            return
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._reply(
                {
                    "ok": False,
                    "operation_id": FIELDLAB_OPERATION_ID,
                    "execution_id": "",
                    "error": "FIELDLAB_REQUEST_NOT_JSON",
                }
            )
            return
        response = self.server.dispatch(request)  # type: ignore[attr-defined]
        if not isinstance(response, Mapping):
            response = {
                "ok": False,
                "operation_id": FIELDLAB_OPERATION_ID,
                "execution_id": "",
                "error": "FIELDLAB_SEMANTIC_CARRIER_INTERNAL_ERROR",
            }
        self._reply(response)

    def _reply(self, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        if len(raw) > MAX_MESSAGE_BYTES:
            raw = (
                b'{"ok":false,"operation_id":"fieldlab.semantic.execute.v1",'
                b'"execution_id":"","error":"FIELDLAB_RESPONSE_SIZE_INVALID"}\n'
            )
        self.wfile.write(raw)


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class FieldLabSemanticUnixServer:
    """Dedicated FieldLab local IPC surface inside the existing Hermes process."""

    def __init__(
        self,
        socket_path: Path = DEFAULT_FIELDLAB_SOCKET_PATH,
        dispatch: Callable[[Mapping[str, Any]], dict[str, Any]] = dispatch_fieldlab_semantic_request,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.dispatch = dispatch
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._server is not None:
            return True
        if os.name != "posix":
            LOGGER.warning("FieldLab semantic carrier requires POSIX Unix sockets")
            return False
        parent = self.socket_path.parent
        if not parent.exists() or not parent.is_dir() or parent.is_symlink():
            LOGGER.info(
                "FieldLab semantic carrier disabled: dedicated socket directory unavailable: %s",
                parent,
            )
            return False

        try:
            if self.socket_path.exists() or self.socket_path.is_symlink():
                current = self.socket_path.lstat()
                if not stat.S_ISSOCK(current.st_mode) or current.st_uid != os.geteuid():
                    raise FieldLabSemanticCarrierError("FIELDLAB_SOCKET_PATH_COLLISION")
                self.socket_path.unlink()

            old_umask = os.umask(0o117)  # socket 0660; parent ACL remains deployment-owned.
            try:
                server = _ThreadingUnixServer(
                    str(self.socket_path),
                    _FieldLabSemanticRequestHandler,
                )
            finally:
                os.umask(old_umask)
            server.dispatch = self.dispatch  # type: ignore[attr-defined]
            mode = stat.S_IMODE(self.socket_path.stat().st_mode)
            if mode & 0o007:
                server.server_close()
                self.socket_path.unlink(missing_ok=True)
                raise FieldLabSemanticCarrierError("FIELDLAB_SOCKET_WORLD_ACCESSIBLE")

            thread = threading.Thread(
                target=server.serve_forever,
                name="hermes-fieldlab-semantic-carrier",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            thread.start()
            LOGGER.info("FieldLab semantic carrier listening on %s", self.socket_path)
            return True
        except FieldLabSemanticCarrierError:
            raise
        except OSError as exc:
            LOGGER.info(
                "FieldLab semantic carrier disabled: cannot bind %s: %s",
                self.socket_path,
                exc,
            )
            return False

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self.cleanup_files()

    def cleanup_files(self) -> None:
        try:
            if self.socket_path.exists() or self.socket_path.is_symlink():
                current = self.socket_path.lstat()
                if stat.S_ISSOCK(current.st_mode) and current.st_uid == os.geteuid():
                    self.socket_path.unlink()
        except OSError:
            pass
