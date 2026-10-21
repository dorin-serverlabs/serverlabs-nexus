# ServerLabs Nexus (`serverlabs-nexus`)

**Bare-Metal GPU Commissioning, Redfish BMC Telemetry & Hybrid Local/Cloud Inference Gateway**  
Developed by **GENOTRONIX S.R.L. (d/b/a SERVERLABS)** · Bacău, Romania (EU)  
Website: [https://serverlabs.ai](https://serverlabs.ai) · Platform Architecture: [https://serverlabs.ai/platform](https://serverlabs.ai/platform)

---

## Why We Built `serverlabs-nexus`

At **ServerLabs (GENOTRONIX S.R.L.)**, we assemble and integrate custom multi-GPU servers and workstations (2x to 8x NVIDIA RTX PRO 6000 Blackwell / H100 / L40S) for European engineering teams.

Before shipping any system from our hardware lab, every server undergoes a **72-hour burn-in and commissioning validation**. We built `serverlabs-nexus` (`v0.4.2`) to automate two critical workflows:
1. **Pre-Delivery Hardware & Inference Commissioning (`nexus diag` & `nexus bench`)**:
   - Inspects Linux `/sys/bus/pci/devices` and NVML (`nvidia-smi`) to catch degraded PCIe risers/slots (e.g., a Gen5 `x16` link training down to `x8`), NUMA node mismatches, and volatile ECC memory errors.
   - Polls out-of-band **Redfish / IPMI BMC** sensors (`/redfish/v1/Chassis/1/Thermal` & `/Power`) on Supermicro, Gigabyte, and ASUS server boards to record inlet/exhaust temperatures, fan RPM curves, and watts-per-token efficiency under full vLLM load.
   - Generates a SHA-256 signed **JSON Commissioning Certificate** shipped with the hardware.
2. **On-Prem Appliance Gateway with Hybrid Cloud Bursting (`nexus.gateway`)**:
   - Runs on the delivered appliance to expose an OpenAI-compatible endpoint (`/v1/chat/completions`) with warm prefix-cache affinity routing across local GPUs.
   - When local GPU KV-cache utilization exceeds threshold (`0.88`) or a node enters thermal protection (`>= 84°C`), requests seamlessly burst to **Microsoft Azure OpenAI / Azure AI Foundry** or **Google Cloud Vertex AI** so customer workloads never drop requests during traffic spikes.

---

## Installation & Quick Start

```bash
git clone https://github.com/dorin-serverlabs/serverlabs-nexus.git
cd serverlabs-nexus
pip install -e .
```

### 1. Run PCIe & GPU Topology Diagnostics (`nexus diag`)
```bash
nexus diag --sysfs-root /sys/bus/pci/devices
```

### 2. Validate an Appliance Deployment Profile (`nexus deploy`)
```bash
nexus deploy -f nexus/profiles/vllm_h100_fp8.yaml
```

### 3. Generate a Burn-In Commissioning Certificate (`nexus bench`)
```bash
nexus bench --serial SL-RO-2026-0142 --gpus "2x NVIDIA RTX PRO 6000 96GB" --model Qwen/Qwen2.5-72B-Instruct-AWQ
```

---

## Running the Test Suite

```bash
pytest -v
```

## License

Apache License 2.0 — Copyright © 2026 **GENOTRONIX S.R.L. (d/b/a SERVERLABS)** (CUI `RO48932335`, Reg. No. `J2023019130409`, Calea Republicii 72, Bacău 600325, Romania).
