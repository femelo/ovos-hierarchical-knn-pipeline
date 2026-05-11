import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

# Labels that bypass the registered-intent check and are always matched
_SPECIAL_LABELS = {"ocp:play", "common_query:common_query", "stop:stop"}

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier


class HierarchicalKNNIntentPipeline(ConfidenceMatcherPipeline):
    """
    OVOS intent pipeline backed by HierarchicalPairKNNClassifier.

    Loads a pre-built FAISS index from disk and performs hierarchical KNN
    classification at inference time. Only intents registered by loaded skills
    (via Adapt or Padatious) are considered, plus the built-in special labels
    (ocp:play, common_query:common_query, stop:stop).

    Configuration keys (under intents → ovos_hierarchical_knn_pipeline):
        index_dir      — path to a local index directory produced by build_index.py;
                         when omitted the default pre-built index is downloaded from
                         HuggingFace (fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2)
        hf_repo_id     — HuggingFace repo to download when index_dir is not set
                         (default: fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2)
        hf_cache_dir   — local cache directory for the HuggingFace snapshot
        conf_high      — minimum probability for match_high  (default 0.7)
        conf_medium    — minimum probability for match_medium (default 0.5)
        conf_low       — minimum probability for match_low   (default 0.15)
        renormalize    — re-scale surviving probabilities to sum to 1 (default False)
        ignore_intents — list of intent labels to suppress
        timeout        — bus wait timeout in seconds (default 1)
    """

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ):
        config = (
            config
            or Configuration().get("intents", {}).get("ovos_hierarchical_knn_pipeline")
            or {}
        )
        super().__init__(bus, config)

        index_dir = self.config.get("index_dir")
        if index_dir:
            self.model = HierarchicalPairKNNClassifier.from_disk(index_dir)
            LOG.info(f"Loaded HierarchicalKNN pipeline from: '{index_dir}'")
        else:
            repo_id = self.config.get(
                "hf_repo_id",
                "fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2",
            )
            cache_dir = self.config.get("hf_cache_dir")
            LOG.info(f"Downloading HierarchicalKNN index from HuggingFace: '{repo_id}'")
            self.model = HierarchicalPairKNNClassifier.from_pretrained(
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            LOG.info("HierarchicalKNN index downloaded and loaded.")

        self.intents: List[str] = []
        self.ignore_labels: List[str] = self.config.get("ignore_intents") or []

        self.bus.on("mycroft.ready", self.handle_sync_intents)
        self.bus.on("padatious:register_intent", self.handle_sync_intents)
        self.bus.on("register_intent", self.handle_sync_intents)
        self.bus.on("detach_intent", self.handle_sync_intents)
        self.bus.on("detach_skill", self.handle_sync_intents)

        self._syncing = False

    def _get_adapt_intents(self, timeout: int = 1) -> List[str]:
        msg = Message("intent.service.adapt.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.adapt.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i["name"] for i in res.data["intents"] if i["name"] not in self.ignore_labels]

    def _get_padatious_intents(self, timeout: int = 1) -> List[str]:
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i for i in res.data["intents"] if i not in self.ignore_labels]

    def handle_sync_intents(self, message: Message) -> None:
        if self._syncing:
            return
        self._syncing = True
        time.sleep(3)
        timeout = self.config.get("timeout", 1)
        try:
            self.intents = list(
                set(self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout))
            )
            LOG.debug(f"HierarchicalKNN registered intents: {len(self.intents)}")

            # Restrict L1 search to the domains of loaded skills.
            # Always include the special-label domains so ocp/common_query/stop
            # remain reachable even when no skill explicitly registers them.
            active_domains = {i.split(":")[0] for i in self.intents if ":" in i}
            active_domains |= {label.split(":")[0] for label in _SPECIAL_LABELS}
            self.model.set_active_domains(list(active_domains))
            LOG.debug(f"HierarchicalKNN active domains: {sorted(active_domains)}")
        except RuntimeError:
            pass
        self._syncing = False

    def _match(self, utterance: str) -> Iterable[Tuple[str, str, float]]:
        """Encode utterance, run KNN prediction, filter to registered intents."""
        probs_dict = self.model.predict_proba([utterance])[0]

        allowed = set(self.intents) | _SPECIAL_LABELS
        filtered = {k: v for k, v in probs_dict.items() if k and k in allowed}

        if not filtered:
            LOG.warning("No KNN predictions match registered intents")
            return

        if self.config.get("renormalize"):
            total = sum(filtered.values())
            if total > 0:
                filtered = {k: v / total for k, v in filtered.items()}

        for label, prob in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
            LOG.debug(f"Match candidate: {label} - prob: {prob}")

            skill_id = label.split(":")[0]
            if label == "ocp:play":
                skill_id = "ovos.common_play"
                label = "ovos.common_play.play_search"
            elif label == "common_query:common_query":
                skill_id = "common_query.openvoiceos"
                label = "common_query.question"
            elif label == "stop:stop":
                skill_id = "stop.openvoiceos"
                label = "mycroft.stop"

            yield skill_id, label, float(prob)

    def match_high(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        min_conf = self.config.get("conf_high", 0.7)
        LOG.debug(f"HierarchicalKNN match_high (min_conf={min_conf}): {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            return IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-hierarchical-knn-pipeline",
                utterance=utterances[0],
            )
        return None

    def match_medium(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        min_conf = self.config.get("conf_medium", 0.5)
        LOG.debug(f"HierarchicalKNN match_medium (min_conf={min_conf}): {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            return IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-hierarchical-knn-pipeline",
                utterance=utterances[0],
            )
        return None

    def match_low(
        self, utterances: List[str], lang: str, message: Message
    ) -> Optional[IntentHandlerMatch]:
        min_conf = self.config.get("conf_low", 0.15)
        LOG.debug(f"HierarchicalKNN match_low (min_conf={min_conf}): {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            return IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-hierarchical-knn-pipeline",
                utterance=utterances[0],
            )
        return None

