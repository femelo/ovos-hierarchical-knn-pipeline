"""End-to-end tests for HierarchicalKNNIntentPipeline using ovoscope.

We avoid `ovoscope.pipeline.PipelineHarness` here because its `__enter__`
constructs `_SinkSkill(bus=None)` before the bus exists (upstream bug).  We
instead drive a `MiniCroft` directly with `skill_ids=[]` and capture the
intent-dispatch message that `IntentService._emit_match_message` puts on the
bus when our pipeline returns a match — this is the same signal ovoscope
itself uses, sourced from `ovos-core/ovos_core/intent_services/service.py`.

The classifier loaders (`from_disk`, `from_pretrained`) are patched at class
level so the plugin loads without any real index file or HuggingFace fetch.
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_config.config import Configuration
from ovoscope import get_minicroft

from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier

PIPELINE_ID = "ovos-hierarchical-knn-pipeline"
CONFIG_KEY = "ovos_hierarchical_knn_pipeline"


class _E2EBase(unittest.TestCase):
    """Shared setup: spin up MiniCroft with our pipeline + mocked classifier."""

    extra_config: dict | None = None

    @classmethod
    def setUpClass(cls):
        cls.mock_model = MagicMock()
        cls.mock_model.predict_proba.return_value = [{}]
        cls.mock_model.set_active_domains = MagicMock()

        cls._patches = [
            patch.object(HierarchicalPairKNNClassifier, "from_disk",
                         return_value=cls.mock_model),
            patch.object(HierarchicalPairKNNClassifier, "from_pretrained",
                         return_value=cls.mock_model),
        ]
        for p in cls._patches:
            p.start()

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig_intents_cfg = intents_cfg.get(CONFIG_KEY)
        plugin_cfg = {"index_dir": "/fake/index", "renormalize": False}
        if cls.extra_config:
            plugin_cfg.update(cls.extra_config)
        intents_cfg[CONFIG_KEY] = plugin_cfg

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=60,
        )
        cls.pipeline: HierarchicalKNNIntentPipeline = (
            cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        )
        cls.pipeline.model = cls.mock_model

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig_intents_cfg is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig_intents_cfg
            for p in cls._patches:
                p.stop()

    def setUp(self):
        self.pipeline.intents = []
        self.pipeline.ignore_labels = list(
            (self.extra_config or {}).get("ignore_intents", []) or []
        )
        self.mock_model.predict_proba.return_value = [{}]

    def _set_probs(self, probs: dict):
        self.mock_model.predict_proba.return_value = [probs]

    def _utterance_msg(self, utterance: str,
                       session_pipeline: list[str] | None = None) -> Message:
        ctx = {}
        if session_pipeline is not None:
            sess = Session(session_id="ovoscope-test",
                           pipeline=session_pipeline)
            ctx["session"] = sess.serialize()
        return Message(
            "recognizer_loop:utterance",
            data={"utterances": [utterance], "lang": "en-US"},
            context=ctx,
        )

    def _send_and_capture(self, utterance: str, expected_types: list[str],
                          timeout: float = 5.0,
                          session_pipeline: list[str] | None = None) -> Message | None:
        """Emit a recognizer_loop:utterance and return the first message
        whose msg_type matches any of *expected_types* — or `None` on timeout
        or `complete_intent_failure`."""
        got: list[Message] = []
        done = threading.Event()
        failed = threading.Event()

        def _capture_match(msg):
            got.append(msg)
            done.set()

        def _capture_fail(msg):
            failed.set()
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _capture_match)
        self.mc.bus.on("complete_intent_failure", _capture_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _capture_match)
            self.mc.bus.remove("complete_intent_failure", _capture_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance: str, timeout: float = 2.0,
                         session_pipeline: list[str] | None = None):
        """Assert that the utterance produces `complete_intent_failure`."""
        failed = threading.Event()
        matched = threading.Event()

        def _on_fail(_msg):
            failed.set()

        def _on_any(msg):
            # Any non-fail event with an intent-like payload counts as a match.
            if msg.msg_type == "complete_intent_failure":
                return
            matched.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            failed.wait(timeout=timeout)
        finally:
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        self.assertTrue(
            failed.is_set(),
            f"Expected no match for {utterance!r}, but got no intent_failure.",
        )


class TestRegisteredIntentMatch(_E2EBase):
    def test_high_confidence_dispatches_intent(self):
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs({"skill_a:my.intent": 0.9})
        msg = self._send_and_capture(
            "turn on the lights",
            expected_types=["skill_a:my.intent"],
        )
        self.assertIsNotNone(msg, "expected match message on bus")
        self.assertEqual(msg.msg_type, "skill_a:my.intent")
        self.assertAlmostEqual(msg.data.get("confidence"), 0.9, places=6)
        self.assertEqual(msg.data.get("utterance"), "turn on the lights")

    def test_below_all_thresholds_no_match(self):
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs({"skill_a:my.intent": 0.05})
        self._expect_no_match("turn on the lights")

    def test_unregistered_intent_no_match(self):
        self.pipeline.intents = []
        self._set_probs({"skill_z:other.intent": 0.99})
        self._expect_no_match("anything goes here")


class TestSpecialLabelRouting(_E2EBase):
    """Special labels (ocp/common_query/stop) are gated by the caller's
    session.pipeline. A test session with the matching pipeline ID lets the
    label through; without it the label is filtered out."""

    def test_ocp_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs({"ocp:play": 0.95})
        msg = self._send_and_capture(
            "play some music",
            expected_types=["ovos.common_play.play_search"],
            session_pipeline=["ovos-ocp-pipeline-plugin-high",
                              "ovos-hierarchical-knn-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "ovos.common_play.play_search")

    def test_common_query_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs({"common_query:common_query": 0.8})
        msg = self._send_and_capture(
            "what is the capital of france",
            expected_types=["common_query.question"],
            session_pipeline=["ovos-common-query-pipeline-plugin",
                              "ovos-hierarchical-knn-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "common_query.question")

    def test_stop_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs({"stop:stop": 0.85})
        msg = self._send_and_capture(
            "stop", expected_types=["mycroft.stop"],
            session_pipeline=["ovos-stop-pipeline-plugin-high",
                              "ovos-hierarchical-knn-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "mycroft.stop")

    def test_ocp_filtered_when_pipeline_absent(self):
        """If the session has no ocp pipeline, an `ocp:play` prediction is
        dropped (the downstream OCP service won't be there to handle it)."""
        self.pipeline.intents = []
        self._set_probs({"ocp:play": 0.99})
        self._expect_no_match(
            "play some music",
            session_pipeline=["ovos-hierarchical-knn-pipeline"],
        )

    def test_stop_filtered_when_pipeline_absent(self):
        self.pipeline.intents = []
        self._set_probs({"stop:stop": 0.99})
        self._expect_no_match(
            "stop",
            session_pipeline=["ovos-hierarchical-knn-pipeline"],
        )


class TestMediumConfidence(_E2EBase):
    def test_medium_threshold_window(self):
        # prob 0.55 is below conf_high (0.7) but above conf_medium (0.5).
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs({"skill_a:my.intent": 0.55})
        msg = self._send_and_capture(
            "ambient command",
            expected_types=["skill_a:my.intent"],
        )
        self.assertIsNotNone(msg)
        self.assertAlmostEqual(msg.data.get("confidence"), 0.55, places=6)


class TestIgnoreIntents(_E2EBase):
    extra_config = {"ignore_intents": ["skill_a:my.intent"]}

    def test_ignored_label_not_matched(self):
        self.pipeline.intents = []
        self._set_probs({"skill_a:my.intent": 0.99})
        self._expect_no_match("turn on the lights")


if __name__ == "__main__":
    unittest.main()
