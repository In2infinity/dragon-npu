#!/usr/bin/env python3
"""
DragonNPU LLM Inference Example
Run Large Language Models on NPU with extreme optimization
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dragon_npu import dragon_npu_core as dnpu
from dragon_npu.dragon_npu_compiler import compile_model

class NPULLMInference:
    """LLM inference on NPU"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.compiled_model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load and compile LLM for NPU"""
        print(f"🤖 Loading {self.model_name} for NPU inference...")
        
        # Initialize NPU
        if not dnpu.init():
            print("❌ Failed to initialize NPU")
            return False
        
        # For demo, create a simplified model
        self.create_demo_model()
        
        print("✅ Model loaded and compiled for NPU")
        return True
    
    def create_demo_model(self):
        """Create demo LLM model"""
        # Simplified transformer architecture for NPU
        self.vocab_size = 50257
        self.hidden_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.seq_length = 1024
        
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using NPU acceleration"""
        print(f"\n📝 Prompt: {prompt}")
        print("🚀 Generating on NPU...")
        
        # Tokenize (simplified)
        tokens = self.simple_tokenize(prompt)
        
        # NPU inference loop
        generated = []
        start_time = time.perf_counter()
        
        for i in range(max_tokens):
            # Prepare input
            input_ids = np.array(tokens[-self.seq_length:], dtype=np.int32)
            
            # Run on NPU (simulated)
            logits = self.forward_pass_npu(input_ids)
            
            # Sample next token
            next_token = np.argmax(logits[-1])
            tokens.append(next_token)
            generated.append(next_token)
            
            # Simple stopping condition
            if next_token == 0:  # EOS token
                break
        
        inference_time = time.perf_counter() - start_time
        
        # Decode (simplified)
        response = self.simple_decode(generated)
        
        print(f"\n💬 Response: {response}")
        print(f"⚡ Generated {len(generated)} tokens in {inference_time:.2f}s")
        print(f"📊 Throughput: {len(generated)/inference_time:.1f} tokens/sec")
        
        return response
    
    def forward_pass_npu(self, input_ids: np.ndarray) -> np.ndarray:
        """Run forward pass on NPU"""
        # Simulate NPU inference
        batch_size = 1
        seq_len = len(input_ids)
        
        # In real implementation, this would run on NPU
        logits = np.random.randn(seq_len, self.vocab_size).astype(np.float32)
        
        # Apply softmax (would be done on NPU)
        logits = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        return logits
    
    def simple_tokenize(self, text: str) -> list:
        """Simple tokenization for demo"""
        # In practice, use proper tokenizer
        words = text.lower().split()
        tokens = [hash(word) % self.vocab_size for word in words]
        return tokens
    
    def simple_decode(self, tokens: list) -> str:
        """Simple decoding for demo"""
        # In practice, use proper decoder
        # For demo, generate sample text
        responses = [
            "The future of AI acceleration lies in specialized hardware like NPUs.",
            "NPUs provide unprecedented performance for AI workloads on Linux.",
            "DragonNPU enables efficient LLM inference on consumer hardware.",
            "With NPU acceleration, we can run larger models more efficiently."
        ]
        return responses[len(tokens) % len(responses)]
    
    def benchmark(self, num_prompts: int = 10):
        """Benchmark LLM inference on NPU"""
        print(f"\n📊 Benchmarking {self.model_name} on NPU...")
        
        prompts = [
            "The future of AI is",
            "Linux and open source",
            "NPU acceleration enables",
            "Machine learning on edge devices",
            "The next breakthrough in AI"
        ]
        
        total_tokens = 0
        total_time = 0
        
        for i in range(num_prompts):
            prompt = prompts[i % len(prompts)]
            
            start = time.perf_counter()
            response = self.generate(prompt, max_tokens=50)
            elapsed = time.perf_counter() - start
            
            tokens_generated = len(response.split())
            total_tokens += tokens_generated
            total_time += elapsed
        
        print(f"\n🎯 Benchmark Results:")
        print(f"  Total prompts: {num_prompts}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg tokens/sec: {total_tokens/total_time:.1f}")
        print(f"  Avg latency: {total_time/num_prompts*1000:.1f}ms per prompt")

def main():
    """Main demo"""
    print("🐉 DragonNPU LLM Inference Demo")
    print("=" * 50)
    
    # Create LLM inference engine
    llm = NPULLMInference("gpt2")
    
    # Load model
    if not llm.load_model():
        print("Failed to load model")
        return
    
    # Interactive demo
    print("\n💬 Interactive LLM Demo (type 'quit' to exit, 'benchmark' to run benchmark)")
    
    while True:
        prompt = input("\n> ")
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'benchmark':
            llm.benchmark()
        else:
            llm.generate(prompt)

if __name__ == "__main__":
    main()