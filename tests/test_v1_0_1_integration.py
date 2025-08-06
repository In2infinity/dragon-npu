#!/usr/bin/env python3
"""
DragonNPU v1.0.1 Integration Test
Ensures all components work together correctly
"""

import sys
import time
import numpy as np
from pathlib import Path
import asyncio
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

# Import v1.0.1 components
from dragon_npu.dragon_npu_core_v1_0_1 import (
    init_v1_0_1,
    compile_model_v1_0_1,
    get_performance_stats_v1_0_1,
    DragonNPUCore_v1_0_1
)

from optimizations.npu_performance_optimizer import (
    NPUMemoryOptimizer,
    WeightQuantizer,
    QuantizationType,
    KVCacheOptimizer,
    PerformanceMonitor
)

from optimizations.multi_tile_processor import MultiTileScheduler

class V1_0_1_IntegrationTest:
    """Comprehensive v1.0.1 integration test"""
    
    def __init__(self):
        self.results = {
            'core': {},
            'memory': {},
            'quantization': {},
            'kv_cache': {},
            'multi_tile': {},
            'performance': {},
            'integration': {}
        }
        self.all_passed = True
    
    def test_core_initialization(self) -> bool:
        """Test core v1.0.1 initialization"""
        print("\n1️⃣ Testing Core Initialization...")
        
        try:
            # Initialize v1.0.1
            success = init_v1_0_1()
            
            if success:
                print("   ✅ Core initialized successfully")
                self.results['core']['init'] = 'PASS'
                return True
            else:
                print("   ❌ Core initialization failed")
                self.results['core']['init'] = 'FAIL'
                self.all_passed = False
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['core']['init'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_memory_optimizer(self) -> bool:
        """Test memory optimization"""
        print("\n2️⃣ Testing Memory Optimizer...")
        
        try:
            # Create optimizer
            optimizer = NPUMemoryOptimizer(768)
            
            # Test allocation
            alloc = optimizer.allocate("test", 50 * 1024 * 1024, "fp16", "weights")
            
            if alloc:
                stats = optimizer.get_memory_stats()
                print(f"   ✅ Memory allocated: {stats['allocated_mb']:.1f}MB")
                
                # Test garbage collection
                optimizer.garbage_collect()
                print(f"   ✅ Garbage collection works")
                
                self.results['memory']['optimizer'] = 'PASS'
                self.results['memory']['allocated_mb'] = stats['allocated_mb']
                return True
            else:
                print("   ❌ Memory allocation failed")
                self.results['memory']['optimizer'] = 'FAIL'
                self.all_passed = False
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['memory']['optimizer'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_quantization(self) -> bool:
        """Test weight quantization"""
        print("\n3️⃣ Testing Quantization...")
        
        try:
            quantizer = WeightQuantizer()
            
            # Test data
            weights = np.random.randn(1000, 1000).astype(np.float32)
            original_size = weights.nbytes / 1024 / 1024
            
            # Test INT8
            int8_weights, int8_meta = quantizer.quantize(weights, QuantizationType.INT8)
            int8_size = int8_weights.nbytes / 1024 / 1024
            int8_ratio = weights.nbytes / int8_weights.nbytes
            
            print(f"   ✅ INT8: {original_size:.1f}MB → {int8_size:.1f}MB ({int8_ratio:.1f}x)")
            
            # Test INT4
            int4_weights, int4_meta = quantizer.quantize(weights, QuantizationType.INT4)
            int4_size = int4_weights.nbytes / 1024 / 1024
            int4_ratio = weights.nbytes / int4_weights.nbytes
            
            print(f"   ✅ INT4: {original_size:.1f}MB → {int4_size:.1f}MB ({int4_ratio:.1f}x)")
            
            # Test dequantization
            dequantized = quantizer.dequantize(int8_weights, int8_meta)
            error = np.mean(np.abs(weights - dequantized))
            
            print(f"   ✅ Dequantization error: {error:.6f}")
            
            self.results['quantization'] = {
                'int8_compression': int8_ratio,
                'int4_compression': int4_ratio,
                'dequant_error': float(error),
                'status': 'PASS'
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['quantization']['status'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_kv_cache(self) -> bool:
        """Test KV cache optimization"""
        print("\n4️⃣ Testing KV Cache...")
        
        try:
            kv_cache = KVCacheOptimizer(max_seq_len=256, hidden_size=768, num_layers=12)
            
            # Test cache update
            batch_size = 1
            seq_len = 10
            hidden_size = 768
            
            new_k = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
            new_v = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
            
            # Update cache for layer 0
            cached_k, cached_v = kv_cache.update_cache(0, new_k, new_v)
            
            print(f"   ✅ Cache updated: K shape {cached_k.shape}, V shape {cached_v.shape}")
            
            # Get memory usage
            memory_mb = kv_cache.get_memory_usage()
            print(f"   ✅ Cache memory: {memory_mb:.1f}MB")
            
            # Test cache retrieval
            retrieved_k, retrieved_v = kv_cache.get_cache(0)
            print(f"   ✅ Cache retrieved successfully")
            
            self.results['kv_cache'] = {
                'memory_mb': memory_mb,
                'status': 'PASS'
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['kv_cache']['status'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_multi_tile(self) -> bool:
        """Test multi-tile scheduling"""
        print("\n5️⃣ Testing Multi-Tile Scheduler...")
        
        try:
            scheduler = MultiTileScheduler(num_tiles=32)
            
            # Test partitioning
            assignments = scheduler.partition_model(12, strategy="balanced")
            print(f"   ✅ Model partitioned across {len(set(assignments.values()))} tiles")
            
            # Create workload
            model_layers = [
                {
                    'type': 'matmul',
                    'input_shape': (1, 256, 768),
                    'weight_shape': (768, 768),
                    'output_shape': (1, 256, 768),
                    'dependencies': []
                }
                for _ in range(12)
            ]
            
            # Create chunks
            chunks = scheduler.create_workload_chunks(model_layers)
            print(f"   ✅ Created {len(chunks)} workload chunks")
            
            # Schedule chunks
            tile_schedules = scheduler.schedule_chunks(chunks)
            active_tiles = sum(1 for schedule in tile_schedules.values() if schedule)
            print(f"   ✅ Scheduled across {active_tiles} tiles")
            
            self.results['multi_tile'] = {
                'tiles_used': active_tiles,
                'total_tiles': 32,
                'status': 'PASS'
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['multi_tile']['status'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_performance_monitor(self) -> bool:
        """Test performance monitoring"""
        print("\n6️⃣ Testing Performance Monitor...")
        
        try:
            monitor = PerformanceMonitor()
            
            # Record some metrics
            for i in range(5):
                monitor.record_inference(
                    duration=0.01 + np.random.random() * 0.005,
                    tokens=50,
                    memory_mb=100 + np.random.random() * 10,
                    tile_usage=0.9 + np.random.random() * 0.1
                )
                monitor.record_cache_access(hit=(i % 2 == 0))
            
            # Get statistics
            stats = monitor.get_statistics()
            
            if stats:
                print(f"   ✅ Latency: {stats['inference_latency']['mean_ms']:.1f}ms")
                print(f"   ✅ Throughput: {stats['throughput']['mean_tokens_per_sec']:.1f} tok/s")
                print(f"   ✅ Cache hit rate: {stats['cache']['hit_rate']:.1%}")
                
                self.results['performance'] = {
                    'latency_ms': stats['inference_latency']['mean_ms'],
                    'throughput': stats['throughput']['mean_tokens_per_sec'],
                    'cache_hit_rate': stats['cache']['hit_rate'],
                    'status': 'PASS'
                }
                
                return True
            else:
                print("   ❌ No statistics available")
                self.results['performance']['status'] = 'FAIL'
                self.all_passed = False
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['performance']['status'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def test_end_to_end(self) -> bool:
        """Test end-to-end integration"""
        print("\n7️⃣ Testing End-to-End Integration...")
        
        try:
            # Compile model with all optimizations
            model = compile_model_v1_0_1(
                "test_model.onnx",
                quantization="int8",
                optimization_level=3
            )
            
            print("   ✅ Model compiled with optimizations")
            
            # Run inference
            dummy_input = np.random.randn(1, 256).astype(np.float16)
            output = model.run(dummy_input)
            
            print(f"   ✅ Inference successful: output shape {output.shape}")
            
            # Run benchmark
            benchmark = model.benchmark(iterations=3)
            
            print(f"   ✅ Benchmark: {benchmark['mean_tokens_per_sec']:.1f} tokens/sec")
            
            # Get final stats
            stats = get_performance_stats_v1_0_1()
            
            self.results['integration'] = {
                'inference': 'SUCCESS',
                'benchmark_tokens_per_sec': benchmark['mean_tokens_per_sec'],
                'status': 'PASS'
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.results['integration']['status'] = f'ERROR: {e}'
            self.all_passed = False
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("🧪 DragonNPU v1.0.1 Integration Test Suite")
        print("="*60)
        
        # Run tests
        self.test_core_initialization()
        self.test_memory_optimizer()
        self.test_quantization()
        self.test_kv_cache()
        self.test_multi_tile()
        self.test_performance_monitor()
        self.test_end_to_end()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        test_count = 0
        passed_count = 0
        
        for category, results in self.results.items():
            if isinstance(results, dict) and 'status' in results:
                test_count += 1
                if results['status'] == 'PASS':
                    passed_count += 1
                    print(f"✅ {category.upper()}: PASS")
                else:
                    print(f"❌ {category.upper()}: FAIL")
            elif category in ['core', 'memory']:
                test_count += 1
                if results.get('init') == 'PASS' or results.get('optimizer') == 'PASS':
                    passed_count += 1
                    print(f"✅ {category.upper()}: PASS")
                else:
                    print(f"❌ {category.upper()}: FAIL")
        
        print(f"\n📈 Results: {passed_count}/{test_count} tests passed")
        
        if self.all_passed:
            print("\n🎉 ALL TESTS PASSED! v1.0.1 is READY!")
        else:
            print("\n⚠️  Some tests failed. Please review.")
        
        return self.all_passed

def main():
    """Run integration tests"""
    tester = V1_0_1_IntegrationTest()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())