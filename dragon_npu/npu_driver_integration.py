#!/usr/bin/env python3
"""
NPU Driver Integration Module
Bridges the existing NPU driver implementation with DragonNPU system
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NPUDriverStatus:
    """NPU driver status information"""
    driver_installed: bool = False
    module_loaded: bool = False
    device_present: bool = False
    xrt_available: bool = False
    device_path: str = ""
    pci_device: str = ""
    firmware_version: str = ""
    driver_version: str = ""

@dataclass
class NPUPerformanceMetrics:
    """NPU performance metrics"""
    latency_us: float = 0.0
    throughput_ops_sec: float = 0.0
    memory_usage_mb: float = 0.0
    active_processes: int = 0
    error_count: int = 0
    uptime_seconds: float = 0.0

class NPUDriverIntegration:
    """Integration with the existing NPU driver implementation"""
    
    def __init__(self, npu_base_path: str = "/home/power/Dragonfire/backend/NPU"):
        self.npu_base_path = Path(npu_base_path)
        self.scripts_path = self.npu_base_path / "scripts"
        self.packages_path = self.npu_base_path / "packages"
        self.tests_path = self.npu_base_path / "tests"
        self.src_path = self.npu_base_path / "src"
        
        # Driver paths
        self.xrt_setup_path = Path("/opt/xilinx/xrt/setup.sh")
        self.device_path = Path("/dev/accel/accel0")
        self.module_name = "amdxdna"
        
        # Test binaries
        self.test_binary = self.src_path / "xdna-driver/build/example_build/example_noop_test"
        self.validate_xclbin = self.npu_base_path / "tests/bins/1502_00/validate.xclbin"
        
        self._status = None
        self._metrics = None
    
    def check_driver_status(self) -> NPUDriverStatus:
        """Comprehensive driver status check"""
        status = NPUDriverStatus()
        
        try:
            # Check if packages are installed
            status.driver_installed = self._check_packages_installed()
            
            # Check if kernel module is loaded
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            status.module_loaded = self.module_name in result.stdout
            
            # Check device node
            status.device_present = self.device_path.exists()
            status.device_path = str(self.device_path) if status.device_present else ""
            
            # Check XRT availability
            status.xrt_available = self.xrt_setup_path.exists()
            
            # Get PCI device info
            status.pci_device = self._get_pci_device_info()
            
            # Get versions
            if status.xrt_available:
                status.driver_version = self._get_driver_version()
                status.firmware_version = self._get_firmware_version()
            
            self._status = status
            return status
            
        except Exception as e:
            logger.error(f"Error checking driver status: {e}")
            return status
    
    def install_driver(self) -> bool:
        """Install NPU driver using existing scripts"""
        try:
            install_script = self.scripts_path / "install_npu_driver.sh"
            if not install_script.exists():
                logger.error(f"Install script not found: {install_script}")
                return False
            
            logger.info("Starting NPU driver installation...")
            result = subprocess.run([str(install_script)], 
                                  capture_output=True, text=True, cwd=str(self.scripts_path))
            
            if result.returncode == 0:
                logger.info("NPU driver installed successfully")
                return True
            else:
                logger.error(f"Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False
    
    def test_npu_functionality(self) -> Dict[str, Any]:
        """Test NPU functionality using existing test scripts"""
        test_results = {
            'status': False,
            'tests': {},
            'errors': [],
            'performance': {}
        }
        
        try:
            # Run the main test script
            test_script = self.scripts_path / "test_npu.sh"
            if test_script.exists():
                result = subprocess.run([str(test_script)], 
                                      capture_output=True, text=True, cwd=str(self.scripts_path))
                
                test_results['tests']['main_test'] = {
                    'passed': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
            
            # Run example test if available
            if self.test_binary.exists() and self.validate_xclbin.exists():
                result = subprocess.run([str(self.test_binary), str(self.validate_xclbin)],
                                      capture_output=True, text=True, cwd=self.test_binary.parent)
                
                test_results['tests']['noop_test'] = {
                    'passed': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
                # Extract performance metrics from output
                test_results['performance'] = self._parse_performance_metrics(result.stdout)
            
            # Check overall status
            test_results['status'] = all(test.get('passed', False) for test in test_results['tests'].values())
            
            return test_results
            
        except Exception as e:
            logger.error(f"Testing error: {e}")
            test_results['errors'].append(str(e))
            return test_results
    
    def get_performance_metrics(self) -> NPUPerformanceMetrics:
        """Get current NPU performance metrics"""
        metrics = NPUPerformanceMetrics()
        
        try:
            # Get basic device info
            if self.device_path.exists():
                stat = self.device_path.stat()
                metrics.uptime_seconds = stat.st_mtime
            
            # Try to get more detailed metrics from XRT
            if self.xrt_setup_path.exists():
                env = os.environ.copy()
                env.update(self._get_xrt_env())
                
                result = subprocess.run(['xrt-smi', 'examine'], 
                                      capture_output=True, text=True, env=env)
                
                if result.returncode == 0:
                    metrics = self._parse_xrt_metrics(result.stdout, metrics)
            
            # Get process info
            result = subprocess.run(['lsof', str(self.device_path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                metrics.active_processes = len(result.stdout.strip().split('\n')) - 1  # Subtract header
            
            self._metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return metrics
    
    def run_benchmark(self, iterations: int = 100) -> Dict[str, float]:
        """Run NPU benchmark"""
        if not self.test_binary.exists() or not self.validate_xclbin.exists():
            logger.error("Benchmark binaries not available")
            return {}
        
        try:
            import time
            
            times = []
            env = os.environ.copy()
            env.update(self._get_xrt_env())
            
            # Warmup runs
            for _ in range(10):
                subprocess.run([str(self.test_binary), str(self.validate_xclbin)], 
                             capture_output=True, cwd=self.test_binary.parent, env=env)
            
            # Benchmark runs
            for i in range(iterations):
                start_time = time.perf_counter()
                result = subprocess.run([str(self.test_binary), str(self.validate_xclbin)], 
                                      capture_output=True, cwd=self.test_binary.parent, env=env)
                end_time = time.perf_counter()
                
                if result.returncode == 0:
                    times.append(end_time - start_time)
                else:
                    logger.warning(f"Benchmark iteration {i+1} failed")
            
            if not times:
                return {}
            
            times_ms = [t * 1000 for t in times]
            
            return {
                'mean_ms': float(np.mean(times_ms)),
                'std_ms': float(np.std(times_ms)),
                'min_ms': float(np.min(times_ms)),
                'max_ms': float(np.max(times_ms)),
                'median_ms': float(np.median(times_ms)),
                'ops_per_sec': float(iterations / sum(times)),
                'successful_iterations': len(times),
                'total_iterations': iterations
            }
            
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            return {}
    
    def monitor_npu(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor NPU for specified duration"""
        monitor_script = self.scripts_path / "monitor_npu.sh"
        if not monitor_script.exists():
            logger.error("Monitor script not found")
            return {}
        
        try:
            # Run monitoring script in background for specified duration
            import signal
            import time
            
            proc = subprocess.Popen([str(monitor_script)], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, cwd=str(self.scripts_path))
            
            time.sleep(duration)
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=5)
            
            return {
                'duration': duration,
                'output': stdout,
                'error': stderr,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            return {'error': str(e)}
    
    def get_available_test_binaries(self) -> Dict[str, Path]:
        """Get list of available test binaries"""
        binaries = {}
        
        # Check main test directory
        bins_dir = self.tests_path / "bins"
        if bins_dir.exists():
            for device_dir in bins_dir.iterdir():
                if device_dir.is_dir():
                    for bin_file in device_dir.iterdir():
                        if bin_file.suffix in ['.xclbin', '.elf']:
                            key = f"{device_dir.name}_{bin_file.stem}"
                            binaries[key] = bin_file
        
        # Check tools directory
        tools_bins = self.src_path / "xdna-driver/tools/bins"
        if tools_bins.exists():
            for device_dir in tools_bins.iterdir():
                if device_dir.is_dir():
                    for bin_file in device_dir.iterdir():
                        if bin_file.suffix in ['.xclbin', '.elf']:
                            key = f"tools_{device_dir.name}_{bin_file.stem}"
                            binaries[key] = bin_file
        
        return binaries
    
    def run_custom_test(self, xclbin_path: str, test_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run custom test with specified xclbin"""
        if not self.test_binary.exists():
            return {'error': 'Test binary not available'}
        
        xclbin = Path(xclbin_path)
        if not xclbin.exists():
            return {'error': f'XCLBIN not found: {xclbin_path}'}
        
        try:
            env = os.environ.copy()
            env.update(self._get_xrt_env())
            
            # Add test parameters if provided
            if test_params:
                for key, value in test_params.items():
                    env[f"TEST_{key.upper()}"] = str(value)
            
            result = subprocess.run([str(self.test_binary), str(xclbin)], 
                                  capture_output=True, text=True, 
                                  cwd=self.test_binary.parent, env=env)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'performance': self._parse_performance_metrics(result.stdout)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_packages_installed(self) -> bool:
        """Check if NPU packages are installed"""
        try:
            result = subprocess.run(['dpkg', '-l', 'xrt*'], capture_output=True, text=True)
            return 'xrt' in result.stdout and result.returncode == 0
        except:
            return False
    
    def _get_pci_device_info(self) -> str:
        """Get PCI device information"""
        try:
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'neural' in line.lower() or '17f0' in line:
                    return line.strip()
            return ""
        except:
            return ""
    
    def _get_driver_version(self) -> str:
        """Get driver version"""
        try:
            env = self._get_xrt_env()
            result = subprocess.run(['xrt-smi', '--version'], 
                                  capture_output=True, text=True, env=env)
            return result.stdout.strip()
        except:
            return ""
    
    def _get_firmware_version(self) -> str:
        """Get firmware version"""
        try:
            env = self._get_xrt_env()
            result = subprocess.run(['xrt-smi', 'examine'], 
                                  capture_output=True, text=True, env=env)
            # Parse firmware version from output
            for line in result.stdout.split('\n'):
                if 'firmware' in line.lower():
                    return line.split()[-1] if line.split() else ""
            return ""
        except:
            return ""
    
    def _get_xrt_env(self) -> Dict[str, str]:
        """Get XRT environment variables"""
        env_vars = {}
        if self.xrt_setup_path.exists():
            # Source the setup script and extract environment
            try:
                result = subprocess.run(['/bin/bash', '-c', f'source {self.xrt_setup_path} && env'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if '=' in line and ('XILINX' in line or 'XRT' in line):
                        key, value = line.split('=', 1)
                        env_vars[key] = value
            except:
                pass
        return env_vars
    
    def _parse_performance_metrics(self, output: str) -> Dict[str, float]:
        """Parse performance metrics from test output"""
        metrics = {}
        try:
            for line in output.split('\n'):
                if 'latency' in line.lower() and 'us' in line:
                    # Extract latency in microseconds
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'us' in part and i > 0:
                            metrics['latency_us'] = float(parts[i-1])
                            break
                elif 'throughput' in line.lower() or 'ops' in line.lower():
                    # Extract throughput
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'ops' in part and i > 0:
                            metrics['ops_per_sec'] = float(parts[i-1])
                            break
        except:
            pass
        return metrics
    
    def _parse_xrt_metrics(self, output: str, metrics: NPUPerformanceMetrics) -> NPUPerformanceMetrics:
        """Parse XRT examine output for metrics"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'memory' in line.lower():
                    # Try to extract memory usage
                    pass
                elif 'temperature' in line.lower():
                    # Extract temperature if available
                    pass
        except:
            pass
        return metrics
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        status = self.check_driver_status()
        metrics = self.get_performance_metrics()
        
        return {
            'driver_status': asdict(status),
            'performance_metrics': asdict(metrics),
            'available_tests': list(self.get_available_test_binaries().keys()),
            'paths': {
                'npu_base': str(self.npu_base_path),
                'scripts': str(self.scripts_path),
                'tests': str(self.tests_path),
                'device': str(self.device_path)
            },
            'capabilities': {
                'can_install': self.scripts_path.exists(),
                'can_test': self.test_binary.exists(),
                'can_benchmark': self.test_binary.exists() and self.validate_xclbin.exists(),
                'can_monitor': (self.scripts_path / "monitor_npu.sh").exists()
            }
        }

# Convenience functions for easy integration
def get_npu_integration(npu_path: str = None) -> NPUDriverIntegration:
    """Get NPU driver integration instance"""
    if npu_path is None:
        npu_path = "/home/power/Dragonfire/backend/NPU"
    return NPUDriverIntegration(npu_path)

def quick_status_check() -> Dict[str, Any]:
    """Quick NPU status check"""
    integration = get_npu_integration()
    return integration.get_integration_status()

def install_and_test() -> bool:
    """Install driver and run tests"""
    integration = get_npu_integration()
    
    # Check current status
    status = integration.check_driver_status()
    
    # Install if needed
    if not status.driver_installed:
        logger.info("Installing NPU driver...")
        if not integration.install_driver():
            logger.error("Failed to install NPU driver")
            return False
    
    # Test functionality
    logger.info("Testing NPU functionality...")
    test_results = integration.test_npu_functionality()
    
    return test_results['status']

if __name__ == "__main__":
    print("🐉 NPU Driver Integration")
    print("=" * 50)
    
    integration = get_npu_integration()
    status = integration.get_integration_status()
    
    print(f"Driver Status: {'✅' if status['driver_status']['driver_installed'] else '❌'}")
    print(f"Module Loaded: {'✅' if status['driver_status']['module_loaded'] else '❌'}")
    print(f"Device Present: {'✅' if status['driver_status']['device_present'] else '❌'}")
    print(f"XRT Available: {'✅' if status['driver_status']['xrt_available'] else '❌'}")
    
    if status['driver_status']['pci_device']:
        print(f"PCI Device: {status['driver_status']['pci_device']}")
    
    print(f"\nAvailable Tests: {len(status['available_tests'])}")
    print(f"Can Install: {'✅' if status['capabilities']['can_install'] else '❌'}")
    print(f"Can Test: {'✅' if status['capabilities']['can_test'] else '❌'}")
    print(f"Can Benchmark: {'✅' if status['capabilities']['can_benchmark'] else '❌'}")
    
    # Run a quick test if possible
    if status['capabilities']['can_test']:
        print("\n🧪 Running quick test...")
        test_results = integration.test_npu_functionality()
        print(f"Test Status: {'✅ PASSED' if test_results['status'] else '❌ FAILED'}")
        
        if test_results['performance']:
            perf = test_results['performance']
            if 'latency_us' in perf:
                print(f"Latency: {perf['latency_us']:.2f} μs")
            if 'ops_per_sec' in perf:
                print(f"Throughput: {perf['ops_per_sec']:.0f} ops/sec")