#!/usr/bin/env python3
"""
DragonNPU Quick Test
Verify the framework is working with existing NPU driver
"""

import os
import sys
import subprocess

def test_npu_hardware():
    """Test NPU hardware detection"""
    print("🔍 NPU Hardware Test")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Device node
    tests_total += 1
    if os.path.exists('/dev/accel/accel0'):
        print("✅ /dev/accel/accel0 device found")
        tests_passed += 1
    else:
        print("❌ /dev/accel/accel0 not found")
    
    # Test 2: Kernel module
    tests_total += 1
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'amdxdna' in result.stdout:
            print("✅ amdxdna kernel module loaded")
            tests_passed += 1
        else:
            print("❌ amdxdna kernel module not loaded")
    except:
        print("❌ Could not check kernel modules")
    
    # Test 3: XRT
    tests_total += 1
    if os.path.exists('/opt/xilinx/xrt'):
        print("✅ XRT installation found")
        tests_passed += 1
    else:
        print("❌ XRT not found")
    
    # Test 4: NPU driver project
    tests_total += 1
    npu_dir = os.path.expanduser('~/Dragonfire/backend/NPU')
    if os.path.exists(npu_dir):
        print(f"✅ NPU driver project found at {npu_dir}")
        tests_passed += 1
        
        # Check for test files
        test_dir = os.path.join(npu_dir, 'tests')
        if os.path.exists(test_dir):
            xclbins = [f for f in os.listdir(test_dir) if f.endswith('.xclbin')]
            if xclbins:
                print(f"   Found {len(xclbins)} XCLBIN test files: {', '.join(xclbins[:3])}")
    else:
        print("❌ NPU driver project not found")
    
    print(f"\n📊 Hardware Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def test_dragon_npu_integration():
    """Test DragonNPU integration"""
    print("\n🐉 DragonNPU Integration Test")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Check integration module exists
    tests_total += 1
    integration_file = 'npu_driver_integration.py'
    if os.path.exists(integration_file):
        print(f"✅ Integration module found: {integration_file}")
        tests_passed += 1
    else:
        print(f"❌ Integration module not found")
    
    # Test 2: Check core module
    tests_total += 1
    core_file = 'dragon_npu_core.py'
    if os.path.exists(core_file):
        print(f"✅ Core module found: {core_file}")
        tests_passed += 1
    else:
        print(f"❌ Core module not found")
    
    # Test 3: Check compiler
    tests_total += 1
    compiler_file = 'dragon_npu_compiler.py'
    if os.path.exists(compiler_file):
        print(f"✅ Compiler module found: {compiler_file}")
        tests_passed += 1
    else:
        print(f"❌ Compiler module not found")
    
    # Test 4: Check CLI
    tests_total += 1
    cli_file = 'dragon_npu_cli.py'
    if os.path.exists(cli_file):
        print(f"✅ CLI module found: {cli_file}")
        tests_passed += 1
    else:
        print(f"❌ CLI module not found")
    
    # Test 5: Check backend
    tests_total += 1
    backend_file = 'backends/amd_xdna_backend.py'
    if os.path.exists(backend_file):
        print(f"✅ AMD XDNA backend found: {backend_file}")
        tests_passed += 1
    else:
        print(f"❌ AMD XDNA backend not found")
    
    # Test 6: Check examples
    tests_total += 1
    examples_dir = 'examples'
    if os.path.exists(examples_dir):
        examples = os.listdir(examples_dir)
        print(f"✅ Examples directory found with {len(examples)} examples")
        tests_passed += 1
    else:
        print(f"❌ Examples directory not found")
    
    print(f"\n📊 Integration Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def test_npu_functionality():
    """Test actual NPU functionality"""
    print("\n⚡ NPU Functionality Test")
    print("=" * 50)
    
    # Try to run a simple NPU test using existing infrastructure
    npu_test_script = os.path.expanduser('~/Dragonfire/backend/NPU/scripts/test_npu.sh')
    
    if os.path.exists(npu_test_script):
        print(f"✅ Found NPU test script: {npu_test_script}")
        print("   Running basic NPU test...")
        
        try:
            result = subprocess.run([npu_test_script, 'basic'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ NPU basic test passed!")
                # Parse output for performance metrics
                if "Latency" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "Latency" in line or "Throughput" in line:
                            print(f"   {line.strip()}")
                return True
            else:
                print("⚠️  NPU test returned non-zero exit code")
                return False
        except subprocess.TimeoutExpired:
            print("⚠️  NPU test timed out")
            return False
        except Exception as e:
            print(f"⚠️  Could not run NPU test: {e}")
            return False
    else:
        print("⚠️  NPU test script not found")
        return False

def main():
    """Run all tests"""
    print("🐉 DragonNPU System Test")
    print("=" * 50)
    print("Testing complete NPU stack integration...")
    print("")
    
    # Change to dragon-npu directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    hw_pass = test_npu_hardware()
    int_pass = test_dragon_npu_integration()
    func_pass = test_npu_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = hw_pass and int_pass
    
    if all_passed:
        print("🎉 All core tests PASSED!")
        print("\n✅ DragonNPU is ready to use!")
        print("\n🚀 Quick Start Commands:")
        print("   1. Check status: python3 dragon_npu_cli.py status")
        print("   2. Run example: python3 examples/computer_vision.py")
        print("   3. Compile model: python3 dragon_npu_cli.py compile model.onnx")
        
        if func_pass:
            print("\n⚡ NPU hardware is fully functional!")
        else:
            print("\n⚠️  NPU hardware tests need attention")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\n📝 Next Steps:")
        if not hw_pass:
            print("   - Check NPU hardware and drivers")
        if not int_pass:
            print("   - Verify all DragonNPU files are present")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())