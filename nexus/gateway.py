"""OpenAI-Compatible Local Inference Router, Hybrid Cloud Bursting, and Commissioning Benchmark."""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class EngineEndpoint:
    node_id: str
    engine: str  # vllm | trt-llm | sglang | llama-cpp
    model_id: str
    tensor_parallel: int
    kv_cache_utilization: float
    healthy: bool = True
    base_url: str = "http://127.0.0.1:8000/v1"
    gpu_temp_c: int = 52


@dataclass
class CloudBurstTarget:
    """Cloud failover endpoint (Microsoft Azure OpenAI / Foundry or Google Cloud Vertex AI)."""
    provider: str  # azure-openai | google-vertex-ai | aws-bedrock
    endpoint_url: str
    model_deployment: str
    region: str = "swedencentral"
    enabled: bool = True


class CommissioningReport(BaseModel):
    serial_number: str
    chassis_model: str
    gpu_configuration: str
    engine: str
    model_id: str
    concurrency: int
    total_requests: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    itl_p50_ms: float
    aggregate_tok_per_sec: float
    mean_power_watts: float
    joules_per_token: float
    pcie_lanes_verified: bool = True
    ecc_errors_detected: int = 0
    generated_at_utc: str
    sha256_signature: str


class PrefixAffinityRouter:
    """Routes inference requests across local GPU workers with warm KV-cache prefix affinity
    and automatic hybrid cloud bursting when on-prem GPUs exceed utilization or thermal thresholds.
    """

    def __init__(
        self,
        endpoints: List[EngineEndpoint],
        kv_saturation_threshold: float = 0.88,
        thermal_cutoff_c: int = 84,
        cloud_burst_targets: Optional[List[CloudBurstTarget]] = None,
    ) -> None:
        self.endpoints = list(endpoints)
        self.kv_saturation_threshold = kv_saturation_threshold
        self.thermal_cutoff_c = thermal_cutoff_c
        self.cloud_burst_targets = cloud_burst_targets or []
        self._prefix_affinity_map: Dict[str, str] = {}

    @staticmethod
    def compute_prefix_hash(system_prompt: str, first_user_turn: str = "") -> str:
        raw = f"{system_prompt.strip()}||{first_user_turn[:256].strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _eligible_local(self, model_id: str) -> List[EngineEndpoint]:
        return [
            ep
            for ep in self.endpoints
            if ep.model_id == model_id
            and ep.healthy
            and ep.gpu_temp_c < self.thermal_cutoff_c
        ]

    def select_endpoint(
        self, model_id: str, prefix_hash: Optional[str] = None
    ) -> EngineEndpoint:
        candidates = self._eligible_local(model_id)
        if not candidates:
            raise RuntimeError(f"No healthy GPU workers available for model: {model_id}")

        if prefix_hash and prefix_hash in self._prefix_affinity_map:
            pinned_node = self._prefix_affinity_map[prefix_hash]
            for ep in candidates:
                if (
                    ep.node_id == pinned_node
                    and ep.kv_cache_utilization < self.kv_saturation_threshold
                ):
                    return ep

        candidates.sort(key=lambda ep: ep.kv_cache_utilization)
        chosen = candidates[0]
        if prefix_hash:
            self._prefix_affinity_map[prefix_hash] = chosen.node_id
        return chosen

    def resolve_route(
        self, model_id: str, prefix_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Select local bare-metal worker or burst to Azure OpenAI / Google Vertex AI if saturated."""
        candidates = [
            ep
            for ep in self._eligible_local(model_id)
            if ep.kv_cache_utilization < self.kv_saturation_threshold
        ]
        if candidates:
            ep = self.select_endpoint(model_id, prefix_hash=prefix_hash)
            return {
                "target_type": "on_prem_appliance",
                "node_id": ep.node_id,
                "engine": ep.engine,
                "base_url": ep.base_url,
                "kv_cache_utilization": ep.kv_cache_utilization,
            }

        for burst in self.cloud_burst_targets:
            if burst.enabled:
                return {
                    "target_type": "cloud_burst",
                    "provider": burst.provider,
                    "region": burst.region,
                    "endpoint_url": burst.endpoint_url,
                    "model_deployment": burst.model_deployment,
                }

        raise RuntimeError(
            f"All local GPU workers for {model_id} are saturated and no cloud burst target is enabled"
        )


class CommissioningBenchmark:
    """Generates signed hardware + inference commissioning certificates before server shipment."""

    def __init__(
        self,
        serial_number: str,
        chassis_model: str,
        gpu_configuration: str,
        engine: str = "vllm",
        model_id: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    ) -> None:
        self.serial_number = serial_number
        self.chassis_model = chassis_model
        self.gpu_configuration = gpu_configuration
        self.engine = engine
        self.model_id = model_id

    def build_report(
        self,
        ttft_samples_ms: List[float],
        itl_samples_ms: List[float],
        total_tokens_generated: int,
        wall_time_sec: float,
        mean_power_watts: float,
        concurrency: int = 16,
        pcie_lanes_verified: bool = True,
        ecc_errors_detected: int = 0,
    ) -> CommissioningReport:
        if not ttft_samples_ms or not itl_samples_ms or wall_time_sec <= 0:
            raise ValueError("Benchmark samples and positive wall_time_sec are required")

        sorted_ttft = sorted(ttft_samples_ms)
        p95_idx = min(len(sorted_ttft) - 1, int(round(0.95 * (len(sorted_ttft) - 1))))
        ttft_p50 = round(statistics.median(sorted_ttft), 2)
        ttft_p95 = round(sorted_ttft[p95_idx], 2)
        itl_p50 = round(statistics.median(itl_samples_ms), 2)
        tok_per_s = round(total_tokens_generated / wall_time_sec, 2)
        joules_per_tok = round((mean_power_watts * wall_time_sec) / max(1, total_tokens_generated), 4)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload_str = (
            f"{self.serial_number}|{self.chassis_model}|{self.gpu_configuration}|"
            f"{self.engine}|{self.model_id}|{ttft_p50}|{ttft_p95}|{itl_p50}|"
            f"{tok_per_s}|{mean_power_watts}|{ecc_errors_detected}|{ts}"
        )
        sig = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        return CommissioningReport(
            serial_number=self.serial_number,
            chassis_model=self.chassis_model,
            gpu_configuration=self.gpu_configuration,
            engine=self.engine,
            model_id=self.model_id,
            concurrency=concurrency,
            total_requests=len(ttft_samples_ms),
            ttft_p50_ms=ttft_p50,
            ttft_p95_ms=ttft_p95,
            itl_p50_ms=itl_p50,
            aggregate_tok_per_sec=tok_per_s,
            mean_power_watts=mean_power_watts,
            joules_per_token=joules_per_tok,
            pcie_lanes_verified=pcie_lanes_verified,
            ecc_errors_detected=ecc_errors_detected,
            generated_at_utc=ts,
            sha256_signature=sig,
        )
