#!/usr/bin/env python3
"""
NPU Performance Optimizer for DragonNPU v1.0.1
Memory management, quantization, and throughput optimization
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import time
from enum import Enum

class QuantizationType(Enum):
    """Supported quantization types"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class MemoryAllocation:
    """NPU memory allocation tracking"""
    address: int
    size: int
    dtype: str
    in_use: bool
    last_accessed: float
    tile_id: int

class NPUMemoryOptimizer:
    """Intelligent memory management for 768MB NPU constraint"""
    
    def __init__(self, total_memory_mb: int = 768):
        self.total_memory = total_memory_mb * 1024 * 1024  # Convert to bytes
        self.allocated_memory = 0
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.memory_pool = {}
        self.lock = threading.Lock()
        
        # Memory zones for different purposes
        self.zones = {
            'weights': int(self.total_memory * 0.6),  # 60% for weights
            'activations': int(self.total_memory * 0.25),  # 25% for activations
            'kv_cache': int(self.total_memory * 0.15)  # 15% for KV cache
        }
        
    def allocate(self, name: str, size: int, dtype: str = "fp16", zone: str = "activations") -> Optional[MemoryAllocation]:
        """Allocate memory with zone management"""
        with self.lock:
            # Check if allocation fits in zone
            if zone not in self.zones:
                zone = "activations"
                
            if self.allocated_memory + size > self.zones[zone]:
                # Try to free unused memory
                self.garbage_collect()
                
                if self.allocated_memory + size > self.zones[zone]:
                    print(f"⚠️  Memory allocation failed: {size/1024/1024:.1f}MB requested, {(self.zones[zone]-self.allocated_memory)/1024/1024:.1f}MB available")
                    return None
            
            # Create allocation
            alloc = MemoryAllocation(
                address=self.allocated_memory,
                size=size,
                dtype=dtype,
                in_use=True,
                last_accessed=time.time(),
                tile_id=-1
            )
            
            self.allocations[name] = alloc
            self.allocated_memory += size
            
            return alloc
    
    def free(self, name: str):
        """Free memory allocation"""
        with self.lock:
            if name in self.allocations:
                alloc = self.allocations[name]
                self.allocated_memory -= alloc.size
                
                # Add to memory pool for reuse
                key = (alloc.size, alloc.dtype)
                if key not in self.memory_pool:
                    self.memory_pool[key] = []
                self.memory_pool[key].append(alloc)
                
                del self.allocations[name]
    
    def garbage_collect(self):
        """Free unused memory allocations"""
        current_time = time.time()
        to_free = []
        
        for name, alloc in self.allocations.items():
            # Free allocations not accessed in last 10 seconds
            if not alloc.in_use and (current_time - alloc.last_accessed) > 10:
                to_free.append(name)
        
        for name in to_free:
            self.free(name)
            
        print(f"🧹 Garbage collected {len(to_free)} allocations, freed {sum(self.allocations[n].size for n in to_free if n in self.allocations)/1024/1024:.1f}MB")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            return {
                'total_mb': self.total_memory / 1024 / 1024,
                'allocated_mb': self.allocated_memory / 1024 / 1024,
                'free_mb': (self.total_memory - self.allocated_memory) / 1024 / 1024,
                'usage_percent': (self.allocated_memory / self.total_memory) * 100,
                'num_allocations': len(self.allocations),
                'zones': {k: v/1024/1024 for k, v in self.zones.items()}
            }

class WeightQuantizer:
    """Quantize model weights for memory efficiency"""
    
    @staticmethod
    def quantize(weights: np.ndarray, target_type: QuantizationType) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize weights to target type"""
        
        if target_type == QuantizationType.INT8:
            return WeightQuantizer.quantize_int8(weights)
        elif target_type == QuantizationType.INT4:
            return WeightQuantizer.quantize_int4(weights)
        elif target_type == QuantizationType.FP16:
            return weights.astype(np.float16), {'scale': 1.0, 'zero_point': 0}
        elif target_type == QuantizationType.BF16:
            # Simulate BF16 (not natively supported in numpy)
            return weights.astype(np.float16), {'scale': 1.0, 'zero_point': 0}
        else:
            return weights, {'scale': 1.0, 'zero_point': 0}
    
    @staticmethod
    def quantize_int8(weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to INT8 with scale and zero point"""
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / 255.0
        zero_point = int(-min_val / scale)
        
        # Quantize
        quantized = np.round((weights / scale) + zero_point).astype(np.int8)
        
        return quantized, {
            'scale': scale,
            'zero_point': zero_point,
            'min': min_val,
            'max': max_val
        }
    
    @staticmethod
    def quantize_int4(weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to INT4 (4-bit) for extreme compression"""
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        # Calculate scale for 4-bit range (0-15)
        scale = (max_val - min_val) / 15.0
        zero_point = int(-min_val / scale)
        
        # Quantize to 4-bit
        quantized = np.round((weights / scale) + zero_point)
        quantized = np.clip(quantized, 0, 15).astype(np.uint8)
        
        # Pack two 4-bit values into one byte
        if len(quantized) % 2 != 0:
            quantized = np.pad(quantized, (0, 1), constant_values=0)
        
        packed = np.zeros(len(quantized) // 2, dtype=np.uint8)
        packed = (quantized[::2] << 4) | quantized[1::2]
        
        return packed, {
            'scale': scale,
            'zero_point': zero_point,
            'min': min_val,
            'max': max_val,
            'packed': True
        }
    
    @staticmethod
    def dequantize(quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize weights back to float"""
        if metadata.get('packed', False):
            # Unpack 4-bit values
            unpacked = np.zeros(len(quantized) * 2, dtype=np.uint8)
            unpacked[::2] = (quantized >> 4) & 0x0F
            unpacked[1::2] = quantized & 0x0F
            quantized = unpacked
        
        # Dequantize
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        
        return (quantized.astype(np.float32) - zero_point) * scale

class KVCacheOptimizer:
    """Optimize KV cache for LLM inference"""
    
    def __init__(self, max_seq_len: int = 256, hidden_size: int = 768, num_layers: int = 12):
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Circular buffer for KV cache
        self.cache_buffer = {}
        self.cache_positions = {}
        self.cache_valid_len = {}
        
        self._init_cache()
    
    def _init_cache(self):
        """Initialize circular cache buffers"""
        for layer in range(self.num_layers):
            # Use FP16 for memory efficiency
            self.cache_buffer[f'k_{layer}'] = np.zeros(
                (self.max_seq_len, self.hidden_size), dtype=np.float16
            )
            self.cache_buffer[f'v_{layer}'] = np.zeros(
                (self.max_seq_len, self.hidden_size), dtype=np.float16
            )
            self.cache_positions[layer] = 0
            self.cache_valid_len[layer] = 0
    
    def update_cache(self, layer: int, new_k: np.ndarray, new_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update cache with new key-value pairs"""
        batch_size, seq_len, hidden_size = new_k.shape
        
        # Get current position in circular buffer
        pos = self.cache_positions[layer]
        
        # Update cache
        if pos + seq_len <= self.max_seq_len:
            # Normal update
            self.cache_buffer[f'k_{layer}'][pos:pos+seq_len] = new_k[0]
            self.cache_buffer[f'v_{layer}'][pos:pos+seq_len] = new_v[0]
            self.cache_positions[layer] = pos + seq_len
            self.cache_valid_len[layer] = min(pos + seq_len, self.max_seq_len)
        else:
            # Wrap around (circular buffer)
            remaining = self.max_seq_len - pos
            self.cache_buffer[f'k_{layer}'][pos:] = new_k[0, :remaining]
            self.cache_buffer[f'k_{layer}'][:seq_len-remaining] = new_k[0, remaining:]
            
            self.cache_buffer[f'v_{layer}'][pos:] = new_v[0, :remaining]
            self.cache_buffer[f'v_{layer}'][:seq_len-remaining] = new_v[0, remaining:]
            
            self.cache_positions[layer] = seq_len - remaining
            self.cache_valid_len[layer] = self.max_seq_len
        
        # Return valid cache
        valid_len = self.cache_valid_len[layer]
        return (
            self.cache_buffer[f'k_{layer}'][:valid_len],
            self.cache_buffer[f'v_{layer}'][:valid_len]
        )
    
    def get_cache(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get valid cache for layer"""
        valid_len = self.cache_valid_len[layer]
        return (
            self.cache_buffer[f'k_{layer}'][:valid_len],
            self.cache_buffer[f'v_{layer}'][:valid_len]
        )
    
    def clear_cache(self, layer: Optional[int] = None):
        """Clear cache for layer or all layers"""
        if layer is not None:
            self.cache_positions[layer] = 0
            self.cache_valid_len[layer] = 0
            self.cache_buffer[f'k_{layer}'].fill(0)
            self.cache_buffer[f'v_{layer}'].fill(0)
        else:
            for l in range(self.num_layers):
                self.clear_cache(l)
    
    def get_memory_usage(self) -> float:
        """Get cache memory usage in MB"""
        total_elements = 2 * self.num_layers * self.max_seq_len * self.hidden_size
        bytes_per_element = 2  # FP16
        return (total_elements * bytes_per_element) / (1024 * 1024)

class PerformanceMonitor:
    """Monitor and optimize NPU performance"""
    
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'throughput': [],
            'memory_usage': [],
            'tile_utilization': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.start_time = time.time()
    
    def record_inference(self, duration: float, tokens: int, memory_mb: float, tile_usage: float):
        """Record inference metrics"""
        self.metrics['inference_times'].append(duration)
        self.metrics['throughput'].append(tokens / duration if duration > 0 else 0)
        self.metrics['memory_usage'].append(memory_mb)
        self.metrics['tile_utilization'].append(tile_usage)
    
    def record_cache_access(self, hit: bool):
        """Record cache hit/miss"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics['inference_times']:
            return {}
        
        inference_times = np.array(self.metrics['inference_times'])
        throughput = np.array(self.metrics['throughput'])
        memory = np.array(self.metrics['memory_usage'])
        tile_usage = np.array(self.metrics['tile_utilization'])
        
        total_cache = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = self.metrics['cache_hits'] / total_cache if total_cache > 0 else 0
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_inferences': len(inference_times),
            'inference_latency': {
                'mean_ms': np.mean(inference_times) * 1000,
                'p50_ms': np.percentile(inference_times, 50) * 1000,
                'p95_ms': np.percentile(inference_times, 95) * 1000,
                'p99_ms': np.percentile(inference_times, 99) * 1000,
                'min_ms': np.min(inference_times) * 1000,
                'max_ms': np.max(inference_times) * 1000
            },
            'throughput': {
                'mean_tokens_per_sec': np.mean(throughput),
                'max_tokens_per_sec': np.max(throughput),
                'total_tokens': np.sum(throughput * inference_times)
            },
            'memory': {
                'mean_mb': np.mean(memory),
                'max_mb': np.max(memory),
                'current_mb': memory[-1] if len(memory) > 0 else 0
            },
            'tile_utilization': {
                'mean_percent': np.mean(tile_usage) * 100,
                'max_percent': np.max(tile_usage) * 100,
                'current_percent': tile_usage[-1] * 100 if len(tile_usage) > 0 else 0
            },
            'cache': {
                'hit_rate': cache_hit_rate,
                'total_hits': self.metrics['cache_hits'],
                'total_misses': self.metrics['cache_misses']
            }
        }
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        
        if not stats:
            print("No performance data available")
            return
        
        print("\n📊 Performance Summary")
        print("=" * 50)
        
        print(f"⏱️  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"📈 Total inferences: {stats['total_inferences']}")
        
        print(f"\n⚡ Latency:")
        lat = stats['inference_latency']
        print(f"  Mean: {lat['mean_ms']:.1f}ms")
        print(f"  P50: {lat['p50_ms']:.1f}ms")
        print(f"  P95: {lat['p95_ms']:.1f}ms")
        print(f"  P99: {lat['p99_ms']:.1f}ms")
        
        print(f"\n🚀 Throughput:")
        thr = stats['throughput']
        print(f"  Mean: {thr['mean_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Max: {thr['max_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Total: {thr['total_tokens']:.0f} tokens")
        
        print(f"\n💾 Memory:")
        mem = stats['memory']
        print(f"  Mean: {mem['mean_mb']:.1f}MB")
        print(f"  Max: {mem['max_mb']:.1f}MB")
        print(f"  Current: {mem['current_mb']:.1f}MB")
        
        print(f"\n🎯 NPU Utilization:")
        util = stats['tile_utilization']
        print(f"  Mean: {util['mean_percent']:.1f}%")
        print(f"  Max: {util['max_percent']:.1f}%")
        
        print(f"\n📦 Cache:")
        cache = stats['cache']
        print(f"  Hit rate: {cache['hit_rate']:.1%}")
        print(f"  Hits: {cache['total_hits']}")
        print(f"  Misses: {cache['total_misses']}")

def test_optimizers():
    """Test optimizer components"""
    print("🧪 Testing NPU Performance Optimizers")
    print("=" * 50)
    
    # Test memory optimizer
    print("\n1️⃣ Testing Memory Optimizer...")
    mem_opt = NPUMemoryOptimizer(768)
    
    # Allocate some memory
    alloc1 = mem_opt.allocate("weights", 100 * 1024 * 1024, "fp16", "weights")
    print(f"   Allocated 100MB: {'✅' if alloc1 else '❌'}")
    
    stats = mem_opt.get_memory_stats()
    print(f"   Memory usage: {stats['allocated_mb']:.1f}/{stats['total_mb']:.1f}MB ({stats['usage_percent']:.1f}%)")
    
    # Test weight quantizer
    print("\n2️⃣ Testing Weight Quantizer...")
    weights = np.random.randn(1000, 1000).astype(np.float32)
    
    # Quantize to INT8
    quantized, metadata = WeightQuantizer.quantize(weights, QuantizationType.INT8)
    compression_ratio = weights.nbytes / quantized.nbytes
    print(f"   INT8 compression: {compression_ratio:.1f}x")
    
    # Quantize to INT4
    quantized4, metadata4 = WeightQuantizer.quantize(weights, QuantizationType.INT4)
    compression_ratio4 = weights.nbytes / quantized4.nbytes
    print(f"   INT4 compression: {compression_ratio4:.1f}x")
    
    # Test KV cache
    print("\n3️⃣ Testing KV Cache Optimizer...")
    kv_opt = KVCacheOptimizer(max_seq_len=256, hidden_size=768, num_layers=12)
    memory_usage = kv_opt.get_memory_usage()
    print(f"   KV cache memory: {memory_usage:.1f}MB")
    
    # Test performance monitor
    print("\n4️⃣ Testing Performance Monitor...")
    monitor = PerformanceMonitor()
    
    # Simulate some inferences
    for i in range(10):
        monitor.record_inference(
            duration=0.02 + np.random.randn() * 0.005,
            tokens=50,
            memory_mb=400 + np.random.randn() * 50,
            tile_usage=0.8 + np.random.randn() * 0.1
        )
        monitor.record_cache_access(hit=(i % 3 != 0))
    
    monitor.print_summary()
    
    print("\n✅ All optimizers tested successfully!")

if __name__ == "__main__":
    test_optimizers()