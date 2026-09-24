from __future__ import annotations

import copy
import json
import os
import socket
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gateway import fieldlab_semantic_carrier as carrier


def _capsule() -> dict:
    return {
        "capsule_version": 2,
        "semantic_stage": "PERSON",
        "runtime_identity": {
            "protocol_version": "V2.7",
            "ruleset_id": "8B80A594EA0FE250",
            "runtime_snapshot_sha256": "a" * 64,
            "code_authority_ref": "test",
        },
        "input_capsule_hash": "b" * 64,
        "evidence": {
            "request_context": {
                "date": "2026-09-24",
                "source_text_hash": "c" * 64,
            },
            "person_evidence": {"fixture": "exact"},
        },
        "rule_contract": [{"id": "PERSON.ZIPING.001", "sig": "fixture"}],
        "output_contract": {
            "type": "PersonSemanticInput",
            "required": [
                "status",
                "P_pre",
                "support",
                "avoid",
                "unresolved",
                "primary_need",
            ],
        },
        "forbidden_information": ["revenue", "outcome"],
        "provenance": {"source": "fixture"},
        "authority": False,
    }


def _person_result() -> dict:
    return {
        "status": "READY",
        "P_pre": {"fixture": 1},
        "support": ["x"],
        "avoid": ["y"],
        "unresolved": [],
        "primary_need": "x",
    }


class _FakeAgent:
    def __init__(self, result: dict, *, api_calls: int = 1) -> None:
        self.result = result
        self.api_calls = api_calls
        self.enabled_toolsets = []
        self.tools = {}
        self._session_db = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_manager = None
        self.skip_context_files = True
        self.skip_background_review = True
        self.max_iterations = 1
        self.iteration_budget = SimpleNamespace(max_total=1, used=0)
        self._api_max_retries = 1
        self._fallback_chain = []
        self._fallback_model = None
        self._fallback_index = 0
        self._budget_grace_call = False
        self.compression_enabled = False
        self._persist_disabled = True
        self._cached_system_prompt = carrier.FIELDLAB_SEMANTIC_SYSTEM_PROMPT
        self.ephemeral_system_prompt = None
        self.prompt = None
        self.closed = False

    def run_conversation(self, prompt: str) -> dict:
        self.prompt = prompt
        return {
            "api_calls": self.api_calls,
            "final_response": json.dumps(
                self.result,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

    def close(self) -> None:
        self.closed = True


class FieldLabSemanticCarrierTests(unittest.TestCase):
    def test_semantic_capsule_business_payload_is_passed_to_model_unchanged(self):
        capsule = _capsule()
        before = copy.deepcopy(capsule)
        fake = _FakeAgent(_person_result())
        seen_timeout = []

        def factory(timeout):
            seen_timeout.append(timeout)
            return fake

        result = carrier.execute_semantic_capsule(
            capsule,
            timeout=12.5,
            agent_factory=factory,
        )

        self.assertEqual(result, _person_result())
        self.assertEqual(capsule, before)
        self.assertEqual(json.loads(fake.prompt), before)
        self.assertEqual(seen_timeout, [12.5])
        self.assertTrue(fake.closed)

    def test_operation_local_model_attempt_count_is_exactly_one(self):
        fake = _FakeAgent(_person_result(), api_calls=2)
        with self.assertRaisesRegex(
            carrier.FieldLabSemanticCarrierError,
            "FIELDLAB_MODEL_ATTEMPT_COUNT_INVALID",
        ):
            carrier.execute_semantic_capsule(
                _capsule(),
                timeout=10,
                agent_factory=lambda timeout: fake,
            )

    def test_operation_local_hardening_disables_tools_memory_session_and_fallback(self):
        fake = _FakeAgent(_person_result())
        fake._fallback_chain = ["ambient"]
        fake._fallback_model = "ambient"
        fake.compression_enabled = True
        fake._budget_grace_call = True
        fake._persist_disabled = False

        carrier._harden_agent_runtime(fake)
        carrier._assert_agent_isolation(fake)

        self.assertEqual(fake.enabled_toolsets, [])
        self.assertFalse(fake.tools)
        self.assertIsNone(fake._session_db)
        self.assertFalse(fake._memory_enabled)
        self.assertFalse(fake._user_profile_enabled)
        self.assertEqual(fake._fallback_chain, [])
        self.assertIsNone(fake._fallback_model)
        self.assertFalse(fake.compression_enabled)
        self.assertFalse(fake._budget_grace_call)
        self.assertTrue(fake._persist_disabled)
        for method_name in carrier._FIELDLAB_RECOVERY_METHODS:
            self.assertIs(
                getattr(fake, method_name),
                carrier._deny_fieldlab_recovery,
            )
            self.assertFalse(getattr(fake, method_name)())

    def test_transport_metadata_stays_outside_exact_capsule(self):
        capsule = _capsule()
        captured = []

        def executor(value, *, timeout):
            captured.append((value, timeout))
            return _person_result()

        response = carrier.dispatch_fieldlab_semantic_request(
            {
                "operation_id": carrier.FIELDLAB_OPERATION_ID,
                "execution_id": "fieldlab:test:0001",
                "timeout": 11,
                "capsule": capsule,
            },
            executor=executor,
        )

        self.assertTrue(response["ok"])
        self.assertIs(captured[0][0], capsule)
        self.assertEqual(captured[0][1], 11.0)
        self.assertNotIn("operation_id", capsule)
        self.assertNotIn("execution_id", capsule)
        self.assertNotIn("timeout", capsule)
        self.assertEqual(response["result"], _person_result())

    def test_unknown_transport_metadata_field_fails_closed(self):
        request = {
            "operation_id": carrier.FIELDLAB_OPERATION_ID,
            "execution_id": "fieldlab:test:0002",
            "timeout": 10,
            "capsule": _capsule(),
            "business_hint": "forbidden",
        }
        response = carrier.dispatch_fieldlab_semantic_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "INVALID_REQUEST_SHAPE")

    def test_dedicated_socket_does_not_create_permission_parent(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "not-created" / "semantic.sock"
            server = carrier.FieldLabSemanticUnixServer(
                socket_path=missing,
                dispatch=lambda request: {
                    "ok": True,
                    "operation_id": carrier.FIELDLAB_OPERATION_ID,
                    "execution_id": request.get("execution_id", ""),
                    "result": _person_result(),
                },
            )
            self.assertFalse(server.start())
            self.assertFalse(missing.parent.exists())

    @unittest.skipUnless(os.name == "posix", "Unix socket isolation is POSIX-only")
    def test_dedicated_socket_is_world_inaccessible_and_one_request_per_connection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fieldlab-semantic.sock"
            calls = []

            def dispatch(request):
                calls.append(request)
                return {
                    "ok": True,
                    "operation_id": carrier.FIELDLAB_OPERATION_ID,
                    "execution_id": request["execution_id"],
                    "result": _person_result(),
                }

            server = carrier.FieldLabSemanticUnixServer(
                socket_path=path,
                dispatch=dispatch,
            )
            self.assertTrue(server.start())
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode & 0o007, 0)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(2)
                    client.connect(str(path))
                    payload = {
                        "operation_id": carrier.FIELDLAB_OPERATION_ID,
                        "execution_id": "fieldlab:test:0003",
                        "timeout": 10,
                        "capsule": _capsule(),
                    }
                    client.sendall(
                        json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        + b"\n"
                    )
                    response = json.loads(client.makefile("rb").readline().decode("utf-8"))
                self.assertTrue(response["ok"])
                self.assertEqual(response["result"], _person_result())
                self.assertEqual(calls, [payload])
            finally:
                server.stop()
            self.assertFalse(path.exists())

    def test_default_socket_is_fieldlab_specific_and_not_supply_or_gateway_control(self):
        value = str(carrier.DEFAULT_FIELDLAB_SOCKET_PATH)
        self.assertIn("hermes-fieldlab", value)
        self.assertNotIn("fourstore-supply-core", value)
        self.assertNotEqual(Path(value).name, "gateway.sock")

    def test_module_has_no_supply_business_dependency(self):
        source = Path(carrier.__file__).read_text(encoding="utf-8")
        self.assertNotIn("fourstore_supply", source)
        self.assertNotIn("SUPPLY_HERMES_BRIDGE_SOCKET", source)


if __name__ == "__main__":
    unittest.main()
