"""Bare-metal PCIe/NUMA topology verification, NVML burn-in checks, and Redfish BMC telemetry.

Used in the ServerLabs hardware integration lab (Bacău & Bucharest) during 72-hour server
burn-in and pre-delivery commissioning, as well as runtime thermal guardrails on customer nodes.
"""
from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from pydantic import BaseModel, Field


class PcieLinkStatus(BaseModel):
    bdf: str = Field(..., description="PCIe Bus:Device.Function address, e.g. 0000:41:00.0")
    vendor_device: str = Field(default="10de:2330", description="PCI vendor:device hex ID")
    current_speed_gt: float = Field(..., description="Negotiated transfer rate in GT/s")
    max_speed_gt: float = Field(..., description="Maximum supported transfer rate in GT/s")
    current_width: int = Field(..., description="Negotiated lane width (x1..x16)")
    max_width: int = Field(..., description="Maximum supported lane width (x1..x16)")
    numa_node: int = Field(default=0, description="Linux NUMA node affinity (-1 if unmapped)")

    @property
    def is_width_degraded(self) -> bool:
        return self.current_width < self.max_width

    @property
    def is_speed_degraded(self) -> bool:
        return self.current_speed_gt < self.max_speed_gt


class GpuDeviceNode(BaseModel):
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_mib: int
    memory_used_mib: int = 0
    temperature_c: int = 35
    power_draw_w: float = 65.0
    power_limit_w: float = 700.0
    ecc_uncorrected_volatile: int = 0
    pcie: Optional[PcieLinkStatus] = None

    @property
    def is_thermally_throttling(self) -> bool:
        return self.temperature_c >= 84

    @property
    def has_ecc_fault(self) -> bool:
        return self.ecc_uncorrected_volatile > 0


class BmcTelemetrySnapshot(BaseModel):
    chassis_id: str = "System.Embedded.1"
    inlet_temp_c: float
    exhaust_temp_c: float
    total_power_watts: float
    psu_redundancy_ok: bool = True
    fan_speeds_rpm: Dict[str, int] = Field(default_factory=dict)
    health_status: str = "OK"


def _parse_gt_speed(raw: str) -> float:
    token = raw.strip().split()[0]
    try:
        return float(token)
    except ValueError:
        return 0.0


def inspect_pcie_topology(sysfs_root: str = "/sys/bus/pci/devices") -> List[PcieLinkStatus]:
    """Scan Linux sysfs PCIe devices for NVIDIA (0x10de) or AMD (0x1002) accelerators."""
    root = Path(sysfs_root)
    if not root.exists():
        return []

    results: List[PcieLinkStatus] = []
    for dev_dir in sorted(root.iterdir()):
        vendor_file = dev_dir / "vendor"
        device_file = dev_dir / "device"
        if not vendor_file.exists():
            continue
        vendor = vendor_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
        if vendor not in ("0x10de", "0x1002"):
            continue
        device = (
            device_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
            if device_file.exists()
            else "0x0000"
        )
        cur_w_file = dev_dir / "current_link_width"
        max_w_file = dev_dir / "max_link_width"
        cur_s_file = dev_dir / "current_link_speed"
        max_s_file = dev_dir / "max_link_speed"
        numa_file = dev_dir / "numa_node"

        if not (cur_w_file.exists() and max_w_file.exists()):
            continue

        cur_w = int(cur_w_file.read_text().strip() or "0")
        max_w = int(max_w_file.read_text().strip() or "0")
        if max_w == 0:
            continue

        cur_s = _parse_gt_speed(cur_s_file.read_text()) if cur_s_file.exists() else 0.0
        max_s = _parse_gt_speed(max_s_file.read_text()) if max_s_file.exists() else cur_s
        numa = int(numa_file.read_text().strip()) if numa_file.exists() else 0

        results.append(
            PcieLinkStatus(
                bdf=dev_dir.name,
                vendor_device=f"{vendor.replace('0x', '')}:{device.replace('0x', '')}",
                current_speed_gt=cur_s,
                max_speed_gt=max_s,
                current_width=cur_w,
                max_width=max_w,
                numa_node=numa,
            )
        )
    return results


def find_cross_numa_tp_groups(
    links: List[PcieLinkStatus], tensor_parallel: int = 2
) -> List[Dict[str, Any]]:
    """Detect tensor-parallel GPU groups that span multiple Linux NUMA sockets.

    When tensorParallel >= 2 spans Socket 0 and Socket 1 without NVLink bridging,
    cross-QPI/UPI all-reduce traffic adds substantial tail latency during burn-in.
    """
    if tensor_parallel < 2 or len(links) < tensor_parallel:
        return []
    warnings: List[Dict[str, Any]] = []
    for start in range(0, len(links) - (len(links) % tensor_parallel), tensor_parallel):
        group = links[start : start + tensor_parallel]
        nodes = sorted({g.numa_node for g in group if g.numa_node >= 0})
        if len(nodes) > 1:
            warnings.append(
                {
                    "group_index": start // tensor_parallel,
                    "tensor_parallel": tensor_parallel,
                    "bdfs": [g.bdf for g in group],
                    "numa_nodes": nodes,
                    "warning": "WARN_CROSS_NUMA_TP",
                }
            )
    return warnings


def parse_nvidia_smi_csv(csv_text: str) -> List[GpuDeviceNode]:
    """Parse nvidia-smi CSV output into validated GpuDeviceNode entries."""
    nodes: List[GpuDeviceNode] = []
    reader = csv.reader(io.StringIO(csv_text.strip()))
    for row in reader:
        if len(row) < 9:
            continue
        ecc_raw = row[9].strip() if len(row) > 9 else "0"
        ecc_val = int(ecc_raw) if ecc_raw.isdigit() else 0
        nodes.append(
            GpuDeviceNode(
                index=int(row[0].strip()),
                name=row[1].strip(),
                uuid=row[2].strip(),
                pci_bus_id=row[3].strip(),
                memory_total_mib=int(float(row[4].strip())),
                memory_used_mib=int(float(row[5].strip())),
                temperature_c=int(float(row[6].strip())),
                power_draw_w=float(row[7].strip()),
                power_limit_w=float(row[8].strip()),
                ecc_uncorrected_volatile=ecc_val,
            )
        )
    return nodes


def query_local_gpus() -> List[GpuDeviceNode]:
    """Query local GPUs via nvidia-smi if present; returns empty list on non-GPU hosts."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,temperature.gpu,power.draw,power.limit,ecc.errors.uncorrected.volatile.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        return parse_nvidia_smi_csv(proc.stdout)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def load_appliance_profile(yaml_path: str | Path) -> Dict[str, Any]:
    """Load and validate a ServerLabs Nexus InferenceAppliance YAML profile."""
    path = Path(yaml_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML profile in {path}")
    api_ver = data.get("apiVersion", "")
    kind = data.get("kind", "")
    if api_ver != "nexus.serverlabs.ai/v1alpha2" or kind != "InferenceAppliance":
        raise ValueError(
            f"Unsupported schema {api_ver}/{kind}; expected nexus.serverlabs.ai/v1alpha2 InferenceAppliance"
        )
    spec = data.get("spec", {})
    validate_nvlink_topology(spec.get("resources", spec))
    return data


def validate_nvlink_topology(spec: Dict[str, Any]) -> bool:
    """Validate tensor-parallel degree, PCIe lane requirements, and minimum VRAM footprint."""
    tp = int(spec.get("tensorParallel", 1))
    if tp not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported tensor_parallel degree: {tp}; must be 1, 2, 4, or 8")

    min_lanes = int(spec.get("minPcieLanes", 8))
    if min_lanes not in (4, 8, 16):
        raise ValueError(f"Invalid minPcieLanes={min_lanes}; expected 4, 8, or 16")

    req_vram_gb = float(spec.get("requiredVramPerGpuGb", 0))
    avail_vram_gb = float(spec.get("availableVramPerGpuGb", req_vram_gb))
    if req_vram_gb > 0 and avail_vram_gb < req_vram_gb:
        raise ValueError(
            f"Insufficient per-GPU VRAM: profile requires {req_vram_gb} GB, node provides {avail_vram_gb} GB"
        )
    return True


class RedfishBmcClient:
    """Out-of-band Redfish BMC telemetry client for Supermicro, Gigabyte, ASUS, and Dell iDRAC servers."""

    def __init__(
        self,
        bmc_host: str,
        username: str,
        password: str,
        verify_tls: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = bmc_host.rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = f"https://{self.base_url}"
        self._client = httpx.Client(
            base_url=self.base_url,
            auth=(username, password),
            verify=verify_tls,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def parse_chassis_telemetry(
        self, thermal_json: Dict[str, Any], power_json: Dict[str, Any]
    ) -> BmcTelemetrySnapshot:
        temps = thermal_json.get("Temperatures", [])
        inlet = 24.0
        exhaust = 42.0
        for t in temps:
            name = str(t.get("Name", "")).lower()
            reading = t.get("ReadingCelsius")
            if reading is None:
                continue
            if "inlet" in name or "ambient" in name or "front" in name:
                inlet = float(reading)
            elif "exhaust" in name or "outlet" in name or "rear" in name:
                exhaust = float(reading)

        fans: Dict[str, int] = {}
        for f in thermal_json.get("Fans", []):
            fname = str(f.get("Name") or f.get("FanName") or f"Fan_{len(fans)+1}")
            rpm = f.get("Reading") or f.get("ReadingRPM") or 0
            fans[fname] = int(rpm)

        power_ctrl = power_json.get("PowerControl", [{}])
        watts = float(power_ctrl[0].get("PowerConsumedWatts", 0.0)) if power_ctrl else 0.0
        status_block = thermal_json.get("Status", {})
        health = str(status_block.get("Health", "OK"))

        return BmcTelemetrySnapshot(
            chassis_id=str(thermal_json.get("Id", "Chassis.1")),
            inlet_temp_c=inlet,
            exhaust_temp_c=exhaust,
            total_power_watts=watts,
            fan_speeds_rpm=fans,
            health_status=health,
        )

    def fetch_snapshot(self, chassis_id: str = "1") -> BmcTelemetrySnapshot:
        thermal_resp = self._client.get(f"/redfish/v1/Chassis/{chassis_id}/Thermal")
        thermal_resp.raise_for_status()
        power_resp = self._client.get(f"/redfish/v1/Chassis/{chassis_id}/Power")
        power_resp.raise_for_status()
        return self.parse_chassis_telemetry(thermal_resp.json(), power_resp.json())

    def close(self) -> None:
        self._client.close()
