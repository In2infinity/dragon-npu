"""
DragonNPU - Bringing AI Acceleration to Linux
==============================================

A revolutionary NPU framework that democratizes AI acceleration for Linux users.

Quick Start:
    >>> import dragon_npu as dnpu
    >>> dnpu.init()
    >>> model = dnpu.compile_model("model.onnx")
    >>> output = model.run(input_data)
"""

__version__ = "1.0.1"
__author__ = "DragonNPU Team"

# Core imports
from .dragon_npu_core import (
    init,
    get_capabilities,
    load_model,
    run,
    NPUCapabilities,
    NPUVendor,
    DragonNPUCore,
    DragonNPURuntime,
)

# Compiler imports
from .dragon_npu_compiler import (
    compile_model,
    compile_onnx,
    compile_pytorch,
    compile_tensorflow,
    DragonNPUCompiler,
    CompilerOptions,
)

# Integration imports (optional)
try:
    from .npu_driver_integration import (
        NPUDriverIntegration,
        get_npu_integration,
    )
except ImportError:
    NPUDriverIntegration = None
    get_npu_integration = None

# Import test functions from core
from .dragon_npu_core import (
    test_npu,
    benchmark_npu,
    monitor_npu,
)

__all__ = [
    # Core
    "init",
    "get_capabilities",
    "load_model",
    "run",
    "compile_model",
    # Classes
    "DragonNPUCore",
    "DragonNPURuntime",
    "DragonNPUCompiler",
    "CompilerOptions",
    "NPUDriverIntegration",
    # Enums
    "NPUCapabilities",
    "NPUVendor",
    # Functions
    "compile_onnx",
    "compile_pytorch",
    "compile_tensorflow",
    "test_npu",
    "benchmark_npu",
    "monitor_npu",
]