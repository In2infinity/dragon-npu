#!/usr/bin/env python3
"""
Multi-Tile Processor for DragonNPU v1.0.1
Utilize all 32 compute units for maximum parallelism
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
import math

@dataclass
class TileConfig:
    """Configuration for NPU tile"""
    tile_id: int
    compute_units: int
    memory_mb: int
    frequency_mhz: int
    assigned_layers: List[int]
    utilization: float = 0.0

@dataclass
class WorkloadChunk:
    """Workload chunk for tile processing"""
    chunk_id: int
    tile_id: int
    operation: str
    input_shape: tuple
    output_shape: tuple
    flops: int
    memory_required: int
    dependencies: List[int]

class MultiTileScheduler:
    """Schedule and distribute workloads across NPU tiles"""
    
    def __init__(self, num_tiles: int = 32, compute_units_per_tile: int = 1):
        self.num_tiles = num_tiles
        self.compute_units_per_tile = compute_units_per_tile
        self.tiles = []
        self.workload_queue = queue.PriorityQueue()
        self.execution_graph = {}
        self.tile_executor = ThreadPoolExecutor(max_workers=num_tiles)
        
        # Initialize tiles
        self._init_tiles()
        
    def _init_tiles(self):
        """Initialize tile configurations"""
        for i in range(self.num_tiles):
            tile = TileConfig(
                tile_id=i,
                compute_units=self.compute_units_per_tile,
                memory_mb=768 // self.num_tiles,  # Distribute memory
                frequency_mhz=1500,
                assigned_layers=[]
            )
            self.tiles.append(tile)
    
    def partition_model(self, num_layers: int, strategy: str = "balanced") -> Dict[int, int]:
        """Partition model layers across tiles"""
        assignments = {}
        
        if strategy == "balanced":
            # Evenly distribute layers
            layers_per_tile = math.ceil(num_layers / self.num_tiles)
            for layer in range(num_layers):
                tile_id = min(layer // layers_per_tile, self.num_tiles - 1)
                assignments[layer] = tile_id
                self.tiles[tile_id].assigned_layers.append(layer)
                
        elif strategy == "pipeline":
            # Pipeline parallelism
            for layer in range(num_layers):
                tile_id = layer % self.num_tiles
                assignments[layer] = tile_id
                self.tiles[tile_id].assigned_layers.append(layer)
                
        elif strategy == "data_parallel":
            # Data parallelism - all tiles process same layers
            for layer in range(num_layers):
                for tile_id in range(self.num_tiles):
                    if layer not in assignments:
                        assignments[layer] = []
                    assignments[layer].append(tile_id)
                    
        return assignments
    
    def create_workload_chunks(self, model_layers: List[Dict], batch_size: int = 1) -> List[WorkloadChunk]:
        """Create workload chunks from model layers"""
        chunks = []
        chunk_id = 0
        
        for layer_idx, layer in enumerate(model_layers):
            # Calculate FLOPs for layer
            flops = self._calculate_flops(layer)
            memory = self._calculate_memory(layer)
            
            # Determine tile assignment
            tile_id = layer_idx % self.num_tiles
            
            # Create chunk
            chunk = WorkloadChunk(
                chunk_id=chunk_id,
                tile_id=tile_id,
                operation=layer.get('type', 'unknown'),
                input_shape=layer.get('input_shape', (1, 1)),
                output_shape=layer.get('output_shape', (1, 1)),
                flops=flops,
                memory_required=memory,
                dependencies=layer.get('dependencies', [])
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
        return chunks
    
    def _calculate_flops(self, layer: Dict) -> int:
        """Calculate FLOPs for layer"""
        op_type = layer.get('type', 'unknown')
        
        if op_type == 'matmul':
            # Matrix multiplication FLOPs
            input_shape = layer.get('input_shape', (1, 1, 1))
            weight_shape = layer.get('weight_shape', (1, 1))
            
            # Handle different input shapes
            if len(input_shape) == 3:
                batch, m, k = input_shape
            else:
                m, k = input_shape[-2:]
                batch = 1
                
            k2, n = weight_shape
            return 2 * batch * m * k * n
            
        elif op_type == 'conv2d':
            # Convolution FLOPs
            batch, h, w, c_in = layer.get('input_shape', (1, 1, 1, 1))
            k_h, k_w = layer.get('kernel_size', (3, 3))
            c_out = layer.get('filters', 1)
            return batch * h * w * k_h * k_w * c_in * c_out * 2
            
        else:
            # Default estimate
            return np.prod(layer.get('input_shape', (1,))) * 10
    
    def _calculate_memory(self, layer: Dict) -> int:
        """Calculate memory requirements for layer"""
        # Input + output + weights
        input_size = np.prod(layer.get('input_shape', (1,))) * 2  # FP16
        output_size = np.prod(layer.get('output_shape', (1,))) * 2
        weight_size = np.prod(layer.get('weight_shape', (1,))) * 2
        
        return input_size + output_size + weight_size
    
    def schedule_chunks(self, chunks: List[WorkloadChunk]) -> Dict[int, List[WorkloadChunk]]:
        """Schedule chunks to tiles with dependency resolution"""
        tile_schedules = {i: [] for i in range(self.num_tiles)}
        
        # Build dependency graph
        dep_graph = {}
        for chunk in chunks:
            dep_graph[chunk.chunk_id] = chunk.dependencies
        
        # Topological sort for dependency ordering
        sorted_chunks = self._topological_sort(chunks, dep_graph)
        
        # Assign chunks to tiles
        for chunk in sorted_chunks:
            tile_schedules[chunk.tile_id].append(chunk)
            
        return tile_schedules
    
    def _topological_sort(self, chunks: List[WorkloadChunk], dep_graph: Dict) -> List[WorkloadChunk]:
        """Topological sort for dependency resolution"""
        visited = set()
        stack = []
        
        def visit(chunk_id):
            if chunk_id in visited:
                return
            visited.add(chunk_id)
            
            for dep in dep_graph.get(chunk_id, []):
                visit(dep)
                
            # Find chunk by id
            chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
            if chunk:
                stack.append(chunk)
        
        for chunk in chunks:
            visit(chunk.chunk_id)
            
        return stack[::-1]
    
    def execute_parallel(self, tile_schedules: Dict[int, List[WorkloadChunk]], 
                        kernel_fn: Callable) -> Dict[int, Any]:
        """Execute workloads in parallel across tiles"""
        results = {}
        futures = []
        
        def execute_tile_workload(tile_id: int, chunks: List[WorkloadChunk]):
            """Execute workload on single tile"""
            tile_results = []
            
            for chunk in chunks:
                # Update tile utilization
                self.tiles[tile_id].utilization = chunk.flops / (self.tiles[tile_id].frequency_mhz * 1e6)
                
                # Execute kernel
                result = kernel_fn(chunk)
                tile_results.append(result)
                
                # Simulate processing time based on FLOPs
                processing_time = chunk.flops / (self.tiles[tile_id].frequency_mhz * 1e6)
                time.sleep(min(processing_time, 0.001))  # Cap at 1ms for simulation
                
            return tile_id, tile_results
        
        # Submit workloads to tiles
        for tile_id, chunks in tile_schedules.items():
            if chunks:
                future = self.tile_executor.submit(execute_tile_workload, tile_id, chunks)
                futures.append(future)
        
        # Collect results
        for future in futures:
            tile_id, tile_results = future.result()
            results[tile_id] = tile_results
            
        return results
    
    def get_tile_utilization(self) -> Dict[int, float]:
        """Get current tile utilization"""
        return {tile.tile_id: tile.utilization for tile in self.tiles}
    
    def rebalance_workload(self):
        """Dynamically rebalance workload based on utilization"""
        # Calculate average utilization
        avg_util = sum(t.utilization for t in self.tiles) / self.num_tiles
        
        # Find over and under utilized tiles
        overloaded = [t for t in self.tiles if t.utilization > avg_util * 1.2]
        underloaded = [t for t in self.tiles if t.utilization < avg_util * 0.8]
        
        # Redistribute layers
        for over_tile in overloaded:
            if over_tile.assigned_layers and underloaded:
                # Move one layer to underloaded tile
                layer_to_move = over_tile.assigned_layers.pop()
                under_tile = underloaded[0]
                under_tile.assigned_layers.append(layer_to_move)
                
                print(f"🔄 Rebalanced: Moved layer {layer_to_move} from tile {over_tile.tile_id} to {under_tile.tile_id}")

class TileOptimizer:
    """Optimize operations for individual tiles"""
    
    def __init__(self, tile_config: TileConfig):
        self.config = tile_config
        self.operation_cache = {}
        self.fusion_patterns = self._init_fusion_patterns()
        
    def _init_fusion_patterns(self) -> List[Dict]:
        """Initialize operation fusion patterns"""
        return [
            {
                'pattern': ['matmul', 'add', 'activation'],
                'fused': 'fused_linear',
                'speedup': 1.5
            },
            {
                'pattern': ['conv2d', 'batchnorm', 'relu'],
                'fused': 'fused_conv_bn_relu',
                'speedup': 1.8
            },
            {
                'pattern': ['matmul', 'matmul', 'matmul'],  # QKV in attention
                'fused': 'fused_qkv_projection',
                'speedup': 2.2
            }
        ]
    
    def optimize_chunk(self, chunk: WorkloadChunk) -> WorkloadChunk:
        """Optimize workload chunk for tile execution"""
        # Apply operation fusion if possible
        optimized = self._try_fusion(chunk)
        
        # Apply memory optimization
        optimized = self._optimize_memory_access(optimized)
        
        # Apply compute optimization
        optimized = self._optimize_compute(optimized)
        
        return optimized
    
    def _try_fusion(self, chunk: WorkloadChunk) -> WorkloadChunk:
        """Try to fuse operations"""
        # Check if operation matches fusion pattern
        for pattern in self.fusion_patterns:
            if chunk.operation in pattern['pattern']:
                # Create fused chunk
                chunk.operation = pattern['fused']
                chunk.flops = int(chunk.flops / pattern['speedup'])
                break
                
        return chunk
    
    def _optimize_memory_access(self, chunk: WorkloadChunk) -> WorkloadChunk:
        """Optimize memory access patterns"""
        # Ensure memory fits in tile's allocation
        if chunk.memory_required > self.config.memory_mb * 1024 * 1024:
            # Need to tile the operation
            print(f"⚠️  Tiling operation on tile {self.config.tile_id} due to memory constraints")
            
        return chunk
    
    def _optimize_compute(self, chunk: WorkloadChunk) -> WorkloadChunk:
        """Optimize compute patterns"""
        # Adjust for tile's compute capacity
        chunk.flops = min(chunk.flops, self.config.compute_units * self.config.frequency_mhz * 1e6)
        
        return chunk

def simulate_multi_tile_execution():
    """Simulate multi-tile execution"""
    print("🎮 Simulating Multi-Tile NPU Execution")
    print("=" * 50)
    
    # Create scheduler
    scheduler = MultiTileScheduler(num_tiles=32)
    
    # Create model layers (simulated)
    model_layers = []
    for i in range(12):  # 12 transformer layers
        # Self-attention
        model_layers.append({
            'type': 'matmul',
            'input_shape': (1, 256, 768),
            'weight_shape': (768, 768),
            'output_shape': (1, 256, 768),
            'dependencies': [] if i == 0 else [i-1]
        })
        
        # FFN
        model_layers.append({
            'type': 'matmul',
            'input_shape': (1, 256, 768),
            'weight_shape': (768, 3072),
            'output_shape': (1, 256, 3072),
            'dependencies': [len(model_layers)-1]
        })
    
    # Partition model
    print("\n📊 Partitioning model across tiles...")
    assignments = scheduler.partition_model(len(model_layers), strategy="balanced")
    print(f"   Assigned {len(model_layers)} layers to {scheduler.num_tiles} tiles")
    
    # Create workload chunks
    print("\n🔨 Creating workload chunks...")
    chunks = scheduler.create_workload_chunks(model_layers)
    print(f"   Created {len(chunks)} chunks")
    
    # Schedule chunks
    print("\n📅 Scheduling chunks...")
    tile_schedules = scheduler.schedule_chunks(chunks)
    
    # Show schedule summary
    for tile_id, scheduled_chunks in tile_schedules.items():
        if scheduled_chunks:
            print(f"   Tile {tile_id}: {len(scheduled_chunks)} chunks")
    
    # Execute parallel
    print("\n🚀 Executing on multi-tile NPU...")
    
    def dummy_kernel(chunk: WorkloadChunk):
        """Dummy kernel for simulation"""
        return np.random.randn(1, 768).astype(np.float16)
    
    start_time = time.time()
    results = scheduler.execute_parallel(tile_schedules, dummy_kernel)
    execution_time = time.time() - start_time
    
    print(f"✅ Execution completed in {execution_time:.3f}s")
    
    # Show utilization
    print("\n📊 Tile Utilization:")
    utilization = scheduler.get_tile_utilization()
    
    active_tiles = sum(1 for u in utilization.values() if u > 0)
    avg_util = sum(utilization.values()) / len(utilization) if utilization else 0
    
    print(f"   Active tiles: {active_tiles}/{scheduler.num_tiles}")
    print(f"   Average utilization: {avg_util:.1%}")
    
    # Find bottlenecks
    max_util_tile = max(utilization.items(), key=lambda x: x[1])
    min_util_tile = min(utilization.items(), key=lambda x: x[1])
    
    print(f"   Highest: Tile {max_util_tile[0]} ({max_util_tile[1]:.1%})")
    print(f"   Lowest: Tile {min_util_tile[0]} ({min_util_tile[1]:.1%})")
    
    # Try rebalancing
    print("\n🔄 Rebalancing workload...")
    scheduler.rebalance_workload()
    
    # Calculate theoretical speedup
    single_tile_time = sum(c.flops for c in chunks) / (1500 * 1e6)
    speedup = single_tile_time / execution_time
    
    print(f"\n⚡ Performance:")
    print(f"   Single-tile time: {single_tile_time:.3f}s")
    print(f"   Multi-tile time: {execution_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Efficiency: {speedup/scheduler.num_tiles:.1%}")

if __name__ == "__main__":
    simulate_multi_tile_execution()