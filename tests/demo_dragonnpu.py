#!/usr/bin/env python3
"""
DragonNPU Demo Script
Demonstrates the integrated NPU functionality with real driver support
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import logging

# Import DragonNPU modules
import dragon_npu_core as dnpu
from backends.amd_xdna_backend import XDNARuntime
from npu_driver_integration import get_npu_integration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")

def demo_basic_initialization():
    """Demo basic DragonNPU initialization"""
    print_section_header("🐉 DragonNPU Core Initialization")
    
    try:
        # Initialize DragonNPU
        print("Initializing DragonNPU...")
        success = dnpu.init()
        
        if success:
            print("✅ DragonNPU initialized successfully")
            
            # Get and display capabilities
            caps = dnpu.get_capabilities()
            print(f"\n📊 NPU Capabilities:")
            print(f"  Vendor: {caps.vendor.value}")
            print(f"  Compute Units: {caps.compute_units}")
            print(f"  Memory: {caps.memory_mb} MB")
            print(f"  Max Frequency: {caps.max_frequency_mhz} MHz")
            print(f"  Supported Data Types: {caps.supported_dtypes}")
            print(f"  Driver Installed: {'✅' if caps.driver_installed else '❌'}")
            print(f"  Real Hardware: {'✅' if caps.real_hardware_detected else '❌'}")
            print(f"  Test Binaries Available: {caps.available_test_binaries}")
            
            return True
        else:
            print("❌ Failed to initialize DragonNPU")
            return False
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return False

def demo_driver_status():
    """Demo driver status checking"""
    print_section_header("📊 NPU Driver Status")
    
    try:
        status = dnpu.get_driver_status()
        
        if 'error' in status:
            print(f"❌ Error getting driver status: {status['error']}")
            return False
        
        driver_status = status.get('driver_status', {})
        capabilities = status.get('capabilities', {})
        paths = status.get('paths', {})
        
        print("Driver Information:")
        print(f"  Installed: {'✅' if driver_status.get('driver_installed') else '❌'}")
        print(f"  Module Loaded: {'✅' if driver_status.get('module_loaded') else '❌'}")
        print(f"  Device Present: {'✅' if driver_status.get('device_present') else '❌'}")
        print(f"  XRT Available: {'✅' if driver_status.get('xrt_available') else '❌'}")
        
        if driver_status.get('pci_device'):
            print(f"  PCI Device: {driver_status['pci_device']}")
        
        if driver_status.get('driver_version'):
            print(f"  Driver Version: {driver_status['driver_version']}")
        
        if driver_status.get('firmware_version'):
            print(f"  Firmware Version: {driver_status['firmware_version']}")
        
        print(f"\nPaths:")
        print(f"  Device: {paths.get('device', 'N/A')}")
        print(f"  NPU Base: {paths.get('npu_base', 'N/A')}")
        print(f"  Scripts: {paths.get('scripts', 'N/A')}")
        
        return driver_status.get('driver_installed', False)
        
    except Exception as e:
        logger.error(f"Driver status error: {e}")
        return False

def demo_available_tests():
    """Demo available test enumeration"""
    print_section_header("🧪 Available Tests")
    
    try:
        tests = dnpu.get_available_tests()
        
        print("Test Categories:")
        print(f"  Basic Tests: {tests['basic_tests']}")
        print(f"  Custom Tests Available: {len(tests['custom_tests'])}")
        
        if tests['test_binaries']:
            print(f"\nTest Binaries ({len(tests['test_binaries'])}):")
            for name, path in list(tests['test_binaries'].items())[:10]:  # Show first 10
                print(f"  - {name}: {path}")
            
            if len(tests['test_binaries']) > 10:
                print(f"  ... and {len(tests['test_binaries']) - 10} more")
        
        return True
        
    except Exception as e:
        logger.error(f"Error getting available tests: {e}")
        return False

def demo_npu_testing():
    """Demo NPU testing functionality"""
    print_section_header("🚀 NPU Functionality Testing")
    
    try:
        # Basic test
        print_subsection("Basic NPU Test")
        print("Running basic NPU functionality test...")
        
        test_result = dnpu.test_npu("basic")
        
        if test_result.get('status', False):
            print("✅ Basic NPU test PASSED")
            
            # Show test details
            if 'tests' in test_result:
                for test_name, test_data in test_result['tests'].items():
                    status = "✅ PASSED" if test_data.get('passed', False) else "❌ FAILED"
                    print(f"  {test_name}: {status}")
                    
                    if not test_data.get('passed', False) and test_data.get('error'):
                        print(f"    Error: {test_data['error']}")
            
            # Show performance metrics
            if 'performance' in test_result:
                perf = test_result['performance']
                print(f"\nPerformance Metrics:")
                for metric, value in perf.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.2f}")
                    else:
                        print(f"  {metric}: {value}")
        else:
            print("❌ Basic NPU test FAILED")
            if 'error' in test_result:
                print(f"  Error: {test_result['error']}")
            
            if 'errors' in test_result:
                for error in test_result['errors']:
                    print(f"  Error: {error}")
        
        return test_result.get('status', False)
        
    except Exception as e:
        logger.error(f"Testing error: {e}")
        return False

def demo_performance_monitoring():
    """Demo performance monitoring"""
    print_section_header("📈 Performance Monitoring")
    
    try:
        print("Getting performance statistics...")
        stats = dnpu._runtime.get_stats() if dnpu._runtime else {}
        
        if 'real_driver_metrics' in stats:
            metrics = stats['real_driver_metrics']
            print(f"\nReal Driver Metrics:")
            for key, value in metrics.items():
                if value:
                    if isinstance(value, float) and key.endswith('_us'):
                        print(f"  {key}: {value:.2f}")
                    elif isinstance(value, float) and key.endswith('_sec'):
                        print(f"  {key}: {value:.0f}")
                    else:
                        print(f"  {key}: {value}")
        
        if 'capabilities' in stats:
            caps = stats['capabilities']
            print(f"\nCapabilities Summary:")
            print(f"  Vendor: {caps.get('vendor', 'Unknown')}")
            print(f"  Driver Installed: {'✅' if caps.get('driver_installed') else '❌'}")
            print(f"  Hardware Detected: {'✅' if caps.get('real_hardware_detected') else '❌'}")
        
        # Try to run a quick monitoring session
        print_subsection("Real-time Monitoring (5 seconds)")
        try:
            monitor_result = dnpu.monitor_npu(5)
            if 'error' not in monitor_result:
                print("✅ Monitoring completed successfully")
                print(f"Duration: {monitor_result.get('duration', 'N/A')} seconds")
            else:
                print(f"❌ Monitoring failed: {monitor_result['error']}")
        except Exception as e:
            print(f"⚠️ Monitoring not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance monitoring error: {e}")
        return False

def demo_backend_integration():
    """Demo AMD XDNA backend integration"""
    print_section_header("🔥 AMD XDNA Backend Integration")
    
    try:
        print("Creating AMD XDNA backend runtime...")
        runtime = XDNARuntime()
        
        if runtime.initialize():
            print("✅ AMD XDNA backend initialized successfully")
            
            # Show driver status from backend
            driver_status = runtime.get_driver_status()
            if 'driver_status' in driver_status:
                ds = driver_status['driver_status']
                print(f"\nBackend Driver Status:")
                print(f"  Module Loaded: {'✅' if ds.get('module_loaded') else '❌'}")
                print(f"  Device Present: {'✅' if ds.get('device_present') else '❌'}")
                print(f"  XRT Available: {'✅' if ds.get('xrt_available') else '❌'}")
            
            # Show available test binaries
            test_binaries = runtime.get_available_test_binaries()
            print(f"\nAvailable Test Binaries: {len(test_binaries)}")
            
            # Try to run a real NPU test
            print_subsection("Real NPU Test via Backend")
            test_result = runtime.run_real_npu_test()
            
            if test_result.get('status', False):
                print("✅ Backend NPU test PASSED")
            elif 'error' not in test_result:
                print("❌ Backend NPU test FAILED")
                if 'errors' in test_result:
                    for error in test_result['errors']:
                        print(f"  Error: {error}")
            else:
                print(f"❌ Backend test error: {test_result['error']}")
            
            # Demo kernel simulation
            print_subsection("Kernel Simulation Demo")
            try:
                runtime.load_kernel("demo_vadd", "vector_add", {"size": 1024})
                
                # Create test data
                a = np.random.randn(1024).astype(np.float32)
                b = np.random.randn(1024).astype(np.float32)
                
                print(f"Executing vector addition (size: {len(a)})...")
                result = runtime.execute("demo_vadd", [a, b])
                
                if result:
                    print(f"✅ Kernel execution completed")
                    print(f"Result shape: {result[0].shape}")
                    print(f"Result dtype: {result[0].dtype}")
                    
                    # Verify result
                    expected = a + b
                    if np.allclose(result[0], expected, rtol=1e-5):
                        print("✅ Result verification PASSED")
                    else:
                        print("❌ Result verification FAILED")
                
            except Exception as e:
                print(f"⚠️ Kernel simulation failed: {e}")
            
            return True
        else:
            print("❌ Failed to initialize AMD XDNA backend")
            return False
            
    except Exception as e:
        logger.error(f"Backend integration error: {e}")
        return False

def demo_benchmarking():
    """Demo NPU benchmarking"""
    print_section_header("⚡ NPU Benchmarking")
    
    try:
        print("Running NPU benchmark (50 iterations)...")
        benchmark_result = dnpu.benchmark_npu(50)
        
        if benchmark_result.get('status', False):
            results = benchmark_result.get('results', {})
            if results:
                print("✅ Benchmark completed successfully")
                print(f"\nBenchmark Results:")
                print(f"  Mean Latency: {results.get('mean_ms', 0):.2f} ms")
                print(f"  Std Deviation: {results.get('std_ms', 0):.2f} ms")
                print(f"  Min Latency: {results.get('min_ms', 0):.2f} ms")
                print(f"  Max Latency: {results.get('max_ms', 0):.2f} ms")
                print(f"  Median Latency: {results.get('median_ms', 0):.2f} ms")
                print(f"  Throughput: {results.get('ops_per_sec', 0):.0f} ops/sec")
                print(f"  Success Rate: {results.get('successful_iterations', 0)}/{results.get('total_iterations', 0)}")
            else:
                print("❌ No benchmark results available")
        else:
            print("❌ Benchmark failed")
            if 'error' in benchmark_result:
                print(f"  Error: {benchmark_result['error']}")
        
        return benchmark_result.get('status', False)
        
    except Exception as e:
        logger.error(f"Benchmarking error: {e}")
        return False

def main():
    """Main demo function"""
    print("🐉 DragonNPU Integrated Demo")
    print("Demonstrating NPU driver integration with DragonNPU system")
    print(f"Current working directory: {os.getcwd()}")
    
    results = {}
    
    # Run demo sections
    results['initialization'] = demo_basic_initialization()
    results['driver_status'] = demo_driver_status()
    results['available_tests'] = demo_available_tests()
    results['npu_testing'] = demo_npu_testing()
    results['performance_monitoring'] = demo_performance_monitoring()
    results['backend_integration'] = demo_backend_integration()
    results['benchmarking'] = demo_benchmarking()
    
    # Summary
    print_section_header("📋 Demo Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Demo Results: {passed_tests}/{total_tests} sections completed successfully")
    
    for section, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {section.replace('_', ' ').title()}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 All demo sections completed successfully!")
        print(f"DragonNPU is fully integrated with the NPU driver.")
    else:
        print(f"\n⚠️  Some demo sections failed.")
        print(f"This may be expected if NPU hardware/driver is not available.")
    
    print(f"\nFor more information, check the documentation in:")
    print(f"  - /home/power/Dragonfire/backend/NPU/README.md")
    print(f"  - /home/power/Dragonfire/backend/NPU/SUCCESS_SUMMARY.md")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        sys.exit(1)