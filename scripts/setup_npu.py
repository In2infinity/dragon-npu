#!/usr/bin/env python3
"""
DragonNPU Setup Script
Easy NPU driver installation and setup within DragonNPU context
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DragonNPU modules with fallback handling
try:
    from npu_driver_integration import get_npu_integration, install_and_test
    NPU_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NPU integration not available: {e}")
    NPU_INTEGRATION_AVAILABLE = False
    get_npu_integration = lambda: None
    install_and_test = lambda: False

try:
    from dragon_npu_core import init as init_dragon_npu, get_capabilities, test_npu, get_driver_status
    DRAGON_NPU_CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DragonNPU core not available: {e}")
    DRAGON_NPU_CORE_AVAILABLE = False
    init_dragon_npu = lambda: False
    get_capabilities = lambda: None
    test_npu = lambda *args, **kwargs: {'status': False, 'error': 'DragonNPU core not available'}
    get_driver_status = lambda: {'error': 'DragonNPU core not available'}

class DragonNPUSetup:
    """DragonNPU setup and configuration manager"""
    
    def __init__(self):
        self.integration = get_npu_integration() if NPU_INTEGRATION_AVAILABLE else None
        self.setup_complete = False
        self.fallback_mode = not (NPU_INTEGRATION_AVAILABLE and DRAGON_NPU_CORE_AVAILABLE)
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements for NPU"""
        requirements = {
            'amd_processor': False,
            'kernel_version': False,
            'memory_available': False,
            'permissions': False,
            'npu_base_path': False
        }
        
        try:
            # Check AMD processor
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                requirements['amd_processor'] = 'amd' in cpuinfo and ('ryzen ai' in cpuinfo or 'strix' in cpuinfo)
            
            # Check kernel version
            import platform
            kernel_version = platform.release()
            major, minor = kernel_version.split('.')[:2]
            requirements['kernel_version'] = int(major) >= 6 and int(minor) >= 10
            
            # Check memory
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        mem_kb = int(line.split()[1])
                        requirements['memory_available'] = mem_kb > 2048000  # 2GB
                        break
            
            # Check permissions
            requirements['permissions'] = os.getuid() != 0  # Should not be root
            
            # Check NPU base path
            requirements['npu_base_path'] = self.integration.npu_base_path.exists()
            
        except Exception as e:
            logger.warning(f"Error checking requirements: {e}")
        
        return requirements
    
    def install_dependencies(self) -> bool:
        """Install system dependencies"""
        logger.info("Installing system dependencies...")
        
        dependencies = [
            'build-essential',
            'cmake',
            'git',
            'python3-dev',
            'python3-pip',
            'libboost-all-dev',
            'opencl-headers',
            'libprotobuf-dev',
            'protobuf-compiler',
            'dkms',
            'linux-headers-$(uname -r)',
            'python3-numpy',
            'python3-scipy'
        ]
        
        try:
            import subprocess
            
            # Update package list
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            
            # Install dependencies
            cmd = ['sudo', 'apt', 'install', '-y'] + dependencies
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("System dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def setup_npu_driver(self) -> bool:
        """Setup NPU driver"""
        logger.info("Setting up NPU driver...")
        
        try:
            # Check current status
            status = self.integration.check_driver_status()
            
            if status.driver_installed and status.module_loaded and status.device_present:
                logger.info("NPU driver already installed and working")
                return True
            
            # Install driver
            logger.info("Installing NPU driver...")
            if not self.integration.install_driver():
                logger.error("Failed to install NPU driver")
                return False
            
            # Verify installation
            status = self.integration.check_driver_status()
            if status.driver_installed and status.module_loaded and status.device_present:
                logger.info("NPU driver installed and verified successfully")
                return True
            else:
                logger.error("NPU driver installation verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up NPU driver: {e}")
            return False
    
    def test_npu_functionality(self) -> bool:
        """Test NPU functionality"""
        logger.info("Testing NPU functionality...")
        
        try:
            # Test with integration module
            test_results = self.integration.test_npu_functionality()
            
            if test_results['status']:
                logger.info("NPU functionality test PASSED")
                
                # Show performance metrics if available
                if 'performance' in test_results:
                    perf = test_results['performance']
                    if 'latency_us' in perf:
                        logger.info(f"Latency: {perf['latency_us']:.2f} μs")
                    if 'ops_per_sec' in perf:
                        logger.info(f"Throughput: {perf['ops_per_sec']:.0f} ops/sec")
                
                return True
            else:
                logger.error("NPU functionality test FAILED")
                for test_name, test_data in test_results.get('tests', {}).items():
                    if not test_data.get('passed', False):
                        logger.error(f"  {test_name}: {test_data.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing NPU functionality: {e}")
            return False
    
    def create_config_file(self) -> bool:
        """Create DragonNPU configuration file"""
        logger.info("Creating DragonNPU configuration...")
        
        config_dir = Path.home() / ".config" / "dragonnpu"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / "config.json"
        
        try:
            # Get current status
            status = self.integration.get_integration_status()
            
            config = {
                "version": "1.0",
                "npu_driver": {
                    "vendor": "AMD_XDNA",
                    "installed": status['driver_status']['driver_installed'],
                    "device_path": status['paths']['device'],
                    "npu_base_path": status['paths']['npu_base']
                },
                "capabilities": status['capabilities'],
                "setup_complete": True,
                "last_updated": str(Path.now())
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating config file: {e}")
            return False
    
    def run_full_setup(self, install_deps: bool = True) -> bool:
        """Run complete NPU setup process"""
        logger.info("Starting DragonNPU full setup...")
        
        # Check system requirements
        requirements = self.check_system_requirements()
        logger.info("System requirements check:")
        for req, status in requirements.items():
            logger.info(f"  {req}: {'✅' if status else '❌'}")
        
        if not all(requirements.values()):
            logger.error("System requirements not met")
            if not requirements['amd_processor']:
                logger.error("AMD Ryzen AI processor required")
            if not requirements['kernel_version']:
                logger.error("Kernel 6.10+ required")
            if not requirements['npu_base_path']:
                logger.error(f"NPU base path not found: {self.integration.npu_base_path}")
            return False
        
        # Install dependencies
        if install_deps:
            if not self.install_dependencies():
                logger.error("Failed to install dependencies")
                return False
        
        # Setup NPU driver
        if not self.setup_npu_driver():
            logger.error("Failed to setup NPU driver")
            return False
        
        # Test functionality
        if not self.test_npu_functionality():
            logger.error("NPU functionality test failed")
            return False
        
        # Create config
        if not self.create_config_file():
            logger.error("Failed to create configuration")
            return False
        
        logger.info("✅ DragonNPU setup completed successfully!")
        self.setup_complete = True
        return True
    
    def show_status(self) -> None:
        """Show current NPU status with fallback handling"""
        print("\n🐉 DragonNPU Status")
        print("=" * 50)
        
        # Check module availability first
        print("Module Availability:")
        print(f"  NPU Integration: {'✅' if NPU_INTEGRATION_AVAILABLE else '❌'}")
        print(f"  DragonNPU Core: {'✅' if DRAGON_NPU_CORE_AVAILABLE else '❌'}")
        print(f"  Fallback Mode: {'⚠️  Yes' if self.fallback_mode else '✅ No'}")
        
        if self.fallback_mode:
            print(f"\n⚠️  Running in fallback mode due to missing modules.")
            print(f"Some features may not be available.")
            
            # Try basic system checks
            if self.integration:
                try:
                    status = self.integration.check_driver_status()
                    print(f"\nBasic Driver Check:")
                    print(f"  Driver Installed: {'✅' if status.driver_installed else '❌'}")
                    print(f"  Module Loaded: {'✅' if status.module_loaded else '❌'}")
                    print(f"  Device Present: {'✅' if status.device_present else '❌'}")
                    print(f"  XRT Available: {'✅' if status.xrt_available else '❌'}")
                except Exception as e:
                    print(f"❌ Error checking driver: {e}")
            else:
                print(f"\n❌ NPU integration not available")
            return
        
        try:
            # Full status check with all modules available
            if init_dragon_npu():
                caps = get_capabilities()
                status = get_driver_status()
                
                print(f"\nNPU Information:")
                print(f"  Vendor: {caps.vendor.value}")
                print(f"  Driver Installed: {'✅' if caps.driver_installed else '❌'}")
                print(f"  Hardware Detected: {'✅' if caps.real_hardware_detected else '❌'}")
                print(f"  Test Binaries: {caps.available_test_binaries}")
                
                if 'driver_status' in status:
                    ds = status['driver_status']
                    print(f"  Module Loaded: {'✅' if ds.get('module_loaded') else '❌'}")
                    print(f"  Device Present: {'✅' if ds.get('device_present') else '❌'}")
                    print(f"  XRT Available: {'✅' if ds.get('xrt_available') else '❌'}")
                    
                    if ds.get('pci_device'):
                        print(f"  PCI Device: {ds['pci_device']}")
                
                # Run quick test
                print("\n🧪 Quick Test:")
                test_result = test_npu("basic")
                if test_result.get('status', False):
                    print("✅ NPU test PASSED")
                    if 'performance' in test_result and test_result['performance']:
                        perf = test_result['performance']
                        for key, value in perf.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.2f}")
                else:
                    print("❌ NPU test failed")
                    if 'error' in test_result:
                        print(f"  Error: {test_result['error']}")
            else:
                print("❌ Failed to initialize DragonNPU")
                
        except Exception as e:
            logger.error(f"Error showing status: {e}")
            print(f"❌ Status check failed: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='DragonNPU Setup Tool')
    parser.add_argument('--setup', action='store_true', help='Run full setup')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--test', action='store_true', help='Test NPU functionality')
    parser.add_argument('--install-deps', action='store_true', help='Install system dependencies')
    parser.add_argument('--install-driver', action='store_true', help='Install NPU driver only')
    parser.add_argument('--no-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = DragonNPUSetup()
    
    if args.status:
        setup.show_status()
    elif args.setup:
        success = setup.run_full_setup(install_deps=not args.no_deps)
        sys.exit(0 if success else 1)
    elif args.test:
        success = setup.test_npu_functionality()
        sys.exit(0 if success else 1)
    elif args.install_deps:
        success = setup.install_dependencies()
        sys.exit(0 if success else 1)
    elif args.install_driver:
        success = setup.setup_npu_driver()
        sys.exit(0 if success else 1)
    else:
        # Default: show status
        setup.show_status()

if __name__ == "__main__":
    main()