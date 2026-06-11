# RUHMI Framework AI MCU Compiler

This directory explains how to use `mcu_compile.py` script for compiling and deploying AI models to Renesas RA8P1 Microcontrollers (Cortex-M85 + Ethos-U55 NPU).

You can download mcu_compile.py script from the respective GitHub [here](https://github.com/renesas/ruhmi-framework-mcu/blob/main/scripts/mcu_compile.py) or via wget/Invoke commands in ruhmi_tools directory:

Linux/WSL:
```bash
wget -O mcu_compile.py https://raw.githubusercontent.com/renesas/ruhmi-framework-mcu/main/scripts/mcu_compile.py
```

Windows (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/renesas/ruhmi-framework-mcu/main/scripts/mcu_compile.py" -OutFile "mcu_compile.py"
```


## Overview

`mcu_compile.py` is the unified compilation script that handles the full pipeline — model loading, quantization (FP32 → INT8), C-code generation (MERA 2.0 / CMSIS-NN / Vela), and optional host-based validation. It supports **two input modes**:

| Mode | When to use |
|------|-------------|
| **YAML config** | Reproducible builds, CI pipelines, complex multi-field configs |
| **CLI arguments** | Quick one-off compilations, notebook cells, scripting |

For more details, visit [here](https://github.com/renesas/ruhmi-framework-mcu/tree/main/scripts)


## Configuration Reference (`config.yaml`) — YAML Mode

Below is a complete annotated `config.yaml` with all supported fields. Only `model_path`, `output_dir`, and `target` are **required**; everything else has sensible defaults.

```yaml
# =============================================================================
# Required fields
# =============================================================================
model_path: /path/to/model.tflite   # Path to model file (.tflite, .onnx, .pte) or directory
output_dir: /path/to/output         # Output directory for compiled C-code
target: cpu                         # 'cpu' (CMSIS-NN) or 'npu' (Ethos-U55)

# =============================================================================
# Quantization
# =============================================================================
quantize: false       # Set true to quantize FP32 model to INT8 before deployment
calib_data: ''        # Path to calibration data (.npy file or directory). Empty = random data
calib_num: 5          # Number of calibration samples to generate when calib_data is empty

# =============================================================================
# External memory
# =============================================================================
external: false       # Force external memory mode (NPU: Vela OSPI | CPU: directory naming)
memory_threshold: 0.8 # Model size threshold (MB) for auto-detecting external memory need

# =============================================================================
# MERA / Vela configuration
# =============================================================================
memory_mode: Sram_Only     # 'Sram_Only' or 'Shared_Sram' (NPU only)
optimization: Performance  # 'Performance' or 'Size' (NPU only)
weight_loc: Flash          # Weight storage location: 'Flash' or 'Iram'
suffix: ''                 # Suffix appended to generated C function names (for multi-model apps)

# =============================================================================
# ONNX options
# =============================================================================
onnx_dims: ''   # Freeze dynamic ONNX dimensions, e.g. 'batch=1,width=224,height=224'

# =============================================================================
# Evaluation / Testing
# =============================================================================
ref_data: false        # Generate reference input/output .npy files for on-target testing
x86: false             # Generate x86 pybind11 bindings for manual host testing
host_evaluate: false   # Build and run on host to validate accuracy (CPU only, implies x86)

# =============================================================================
# Output
# =============================================================================
result: ''      # Path for JUnit XML output (leave empty to skip)
verbose: false  # Print detailed output
```

> **Note:** `model_path` and `output_dir` should be **absolute paths** to avoid ambiguity.



## Integration with the Model Zoo

Each model in the Model Zoo (e.g. `vision/image_classification/resnet8/`) includes a `python/config.yaml` pre-configured for that model. The typical workflow is:

1. Edit the `config.yaml` in the model's `python/` directory — set `model_path` and `output_dir` to absolute paths on your system.
2. Activate the compiler venv (`.mera_venv`).
3. Run `mcu_compile.py` in **YAML mode** from the repository root:

```bash
python ruhmi_tools/mcu_compile.py vision/image_classification/resnet8/python/config.yaml
```

Alternatively, use **CLI mode** for a quick one-off compile without editing a config file:

```bash
# CPU path
python ruhmi_tools/mcu_compile.py \
    vision/image_classification/mobilenetv1/python/model/mobilenet_v1_INT8.tflite \
    vision/image_classification/mobilenetv1/embedded_c/src_mcu \
    --cpu

# NPU path (FP32 + calibration data)
python ruhmi_tools/mcu_compile.py \
    vision/image_classification/mobilenetv1/python/model/mobilenet_v1_FP32.tflite \
    vision/image_classification/mobilenetv1/embedded_c/src_mcu_npu \
    --npu --quantize --calib-data calib_data.npy --calib-num 200
```

Refer to each model's README (e.g. [ResNet8 README](../vision/image_classification/resnet8/README.md)) for model-specific compilation instructions.
