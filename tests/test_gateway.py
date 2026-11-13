"""Unit tests for ServerLabs Nexus PCIe topology inspector, Redfish BMC parser,
prefix-affinity router, hybrid cloud bursting, and CLI commands.
"""
from pathlib import Path
import json
import pytest

from nexus.cli import main as cli_main
from nexus.gateway import (
    CloudBurstTarget,
    CommissioningBenchmark,
    EngineEndpoint,
    PrefixAffinityRouter,
)
from nexus.operator import (
    PcieLinkStatus,
    RedfishBmcClient,
    find_cross_numa_tp_groups,
    inspect_pcie_topology,
    load_appliance_profile,
    parse_nvidia_smi_csv,
    validate_nvlink_topology,
)


def test_router_selects_lowest_kv_utilization():
    router = PrefixAffinityRouter([
        EngineEndpoint("gpu-node-01", "vllm", "qwen-2.5-72b", 2, 0.82),
        EngineEndpoint("gpu-node-02", "vllm", "qwen-2.5-72b", 2, 0.41),
    ])
    selected = router.select_endpoint("qwen-2.5-72b")
    assert selected.node_id == "gpu-node-02"


def test_router_honors_warm_prefix_affinity_when_below_saturation():
    ep1 = EngineEndpoint("gpu-node-01", "vllm", "qwen-2.5-72b", 2, 0.30)
    ep2 = EngineEndpoint("gpu-node-02", "vllm", "qwen-2.5-72b", 2, 0.65)
    router = PrefixAffinityRouter([ep1, ep2], kv_saturation_threshold=0.88)

    phash = PrefixAffinityRouter.compute_prefix_hash("You are a coding assistant.")
    first = router.select_endpoint("qwen-2.5-72b", prefix_hash=phash)
    assert first.node_id == "gpu-node-01"

    ep1.kv_cache_utilization = 0.70
    second = router.select_endpoint("qwen-2.5-72b", prefix_hash=phash)
    assert second.node_id == "gpu-node-01"


def test_router_bursts_to_azure_when_local_gpus_saturated():
    ep1 = EngineEndpoint("gpu-node-01", "vllm", "qwen-2.5-72b", 2, 0.93)
    burst = CloudBurstTarget(
        provider="azure-openai",
        endpoint_url="https://serverlabs-eu.openai.azure.com/openai/deployments/gpt-4o",
        model_deployment="gpt-4o",
        region="swedencentral",
    )
    router = PrefixAffinityRouter([ep1], kv_saturation_threshold=0.88, cloud_burst_targets=[burst])
    route = router.resolve_route("qwen-2.5-72b")
    assert route["target_type"] == "cloud_burst"
    assert route["provider"] == "azure-openai"


def test_inspect_pcie_topology_detects_degraded_riser(tmp_path: Path):
    dev_ok = tmp_path / "0000_41_00.0"
    dev_ok.mkdir()
    (dev_ok / "vendor").write_text("0x10de\n")
    (dev_ok / "device").write_text("0x2330\n")
    (dev_ok / "current_link_width").write_text("16\n")
    (dev_ok / "max_link_width").write_text("16\n")
    (dev_ok / "current_link_speed").write_text("32.0 GT/s PCIe\n")
    (dev_ok / "max_link_speed").write_text("32.0 GT/s PCIe\n")
    (dev_ok / "numa_node").write_text("0\n")

    dev_deg = tmp_path / "0000_81_00.0"
    dev_deg.mkdir()
    (dev_deg / "vendor").write_text("0x10de\n")
    (dev_deg / "device").write_text("0x2330\n")
    (dev_deg / "current_link_width").write_text("8\n")
    (dev_deg / "max_link_width").write_text("16\n")
    (dev_deg / "current_link_speed").write_text("16.0 GT/s PCIe\n")
    (dev_deg / "max_link_speed").write_text("32.0 GT/s PCIe\n")
    (dev_deg / "numa_node").write_text("1\n")

    links = inspect_pcie_topology(str(tmp_path))
    assert len(links) == 2
    assert not links[0].is_width_degraded
    assert links[1].is_width_degraded
    assert links[1].is_speed_degraded


def test_parse_nvidia_smi_csv():
    sample = (
        "0, NVIDIA RTX PRO 6000, GPU-1111, 00000000:41:00.0, 97871, 1420, 44, 78.5, 600.0, 0\n"
        "1, NVIDIA RTX PRO 6000, GPU-2222, 00000000:81:00.0, 97871, 1420, 86, 540.2, 600.0, 2\n"
    )
    gpus = parse_nvidia_smi_csv(sample)
    assert len(gpus) == 2
    assert not gpus[0].is_thermally_throttling
    assert gpus[1].is_thermally_throttling
    assert gpus[1].has_ecc_fault


def test_redfish_bmc_telemetry_parser():
    client = RedfishBmcClient("https://10.10.0.120", "admin", "secret")
    thermal = {
        "Id": "System.Embedded.1",
        "Temperatures": [
            {"Name": "Front Inlet Temp", "ReadingCelsius": 23.5},
            {"Name": "Rear Exhaust Temp", "ReadingCelsius": 46.0},
        ],
        "Fans": [
            {"Name": "SYS_FAN1", "ReadingRPM": 6400},
            {"Name": "SYS_FAN2", "ReadingRPM": 6520},
        ],
        "Status": {"Health": "OK"},
    }
    power = {"PowerControl": [{"PowerConsumedWatts": 1240.0}]}
    snap = client.parse_chassis_telemetry(thermal, power)
    client.close()
    assert snap.inlet_temp_c == 23.5
    assert snap.exhaust_temp_c == 46.0
    assert snap.total_power_watts == 1240.0
    assert snap.fan_speeds_rpm["SYS_FAN1"] == 6400


def test_validate_topology_and_load_profile():
    assert validate_nvlink_topology({"tensorParallel": 4, "minPcieLanes": 16}) is True
    with pytest.raises(ValueError):
        validate_nvlink_topology({"tensorParallel": 3})
    with pytest.raises(ValueError):
        validate_nvlink_topology({"tensorParallel": 2, "requiredVramPerGpuGb": 80, "availableVramPerGpuGb": 48})

    repo_root = Path(__file__).resolve().parent.parent
    profile = load_appliance_profile(repo_root / "nexus" / "profiles" / "vllm_h100_fp8.yaml")
    assert profile["kind"] == "InferenceAppliance"


def test_commissioning_benchmark_signature_and_cli(capsys):
    rc = cli_main(["bench", "--serial", "SL-RO-2026-0199"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["serial_number"] == "SL-RO-2026-0199"
    assert len(out["sha256_signature"]) == 64


def test_cross_numa_tp_detection_and_cli_diag(tmp_path: Path, capsys):
    for idx, (bdf, numa) in enumerate([("0000_21_00.0", "0"), ("0000_a1_00.0", "1")]):
        d = tmp_path / bdf
        d.mkdir()
        (d / "vendor").write_text("0x10de\n")
        (d / "device").write_text("0x2684\n")
        (d / "current_link_width").write_text("16\n")
        (d / "max_link_width").write_text("16\n")
        (d / "current_link_speed").write_text("32.0 GT/s PCIe\n")
        (d / "max_link_speed").write_text("32.0 GT/s PCIe\n")
        (d / "numa_node").write_text(f"{numa}\n")

    links = inspect_pcie_topology(str(tmp_path))
    warnings = find_cross_numa_tp_groups(links, tensor_parallel=2)
    assert len(warnings) == 1
    assert warnings[0]["warning"] == "WARN_CROSS_NUMA_TP"
    assert warnings[0]["numa_nodes"] == [0, 1]

    rc = cli_main(["diag", "--sysfs-root", str(tmp_path), "--tensor-parallel", "2"])
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "WARN_CROSS_NUMA_TP"
    assert len(report["cross_numa_warnings"]) == 1

    # Single-socket placement passes cleanly
    same_socket = [
        PcieLinkStatus(bdf="0000:21:00.0", current_speed_gt=32.0, max_speed_gt=32.0, current_width=16, max_width=16, numa_node=0),
        PcieLinkStatus(bdf="0000:41:00.0", current_speed_gt=32.0, max_speed_gt=32.0, current_width=16, max_width=16, numa_node=0),
    ]
    assert find_cross_numa_tp_groups(same_socket, tensor_parallel=2) == []
