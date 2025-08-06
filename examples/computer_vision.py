#!/usr/bin/env python3
"""
DragonNPU Computer Vision Example
Real-time image processing and inference on NPU
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import dragon_npu_core as dnpu

class NPUVisionProcessor:
    """Computer vision processing on NPU"""
    
    def __init__(self):
        self.models = {}
        self.initialized = False
        
    def initialize(self):
        """Initialize NPU for vision tasks"""
        print("🎯 Initializing NPU for Computer Vision...")
        
        if not dnpu.init():
            print("❌ Failed to initialize NPU")
            return False
        
        # Get NPU capabilities
        caps = dnpu.get_capabilities()
        print(f"✅ NPU Ready: {caps.vendor.value}")
        print(f"   Compute Units: {caps.compute_units}")
        print(f"   Memory: {caps.memory_mb} MB")
        
        self.initialized = True
        return True
    
    def image_classification(self, image: np.ndarray) -> dict:
        """Classify image using NPU-accelerated model"""
        print("\n🖼️ Image Classification on NPU")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # NPU inference
        start = time.perf_counter()
        features = self.extract_features_npu(processed)
        logits = self.classify_npu(features)
        inference_time = (time.perf_counter() - start) * 1000
        
        # Get top predictions
        top_k = self.get_top_k_predictions(logits, k=5)
        
        print(f"⚡ Inference time: {inference_time:.2f}ms")
        print(f"📊 Top predictions:")
        for i, (class_name, confidence) in enumerate(top_k, 1):
            print(f"   {i}. {class_name}: {confidence:.2%}")
        
        return {
            'predictions': top_k,
            'inference_time_ms': inference_time
        }
    
    def object_detection(self, image: np.ndarray) -> list:
        """Detect objects using NPU acceleration"""
        print("\n🎯 Object Detection on NPU")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # NPU inference
        start = time.perf_counter()
        detections = self.detect_objects_npu(processed)
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"⚡ Inference time: {inference_time:.2f}ms")
        print(f"📦 Detected {len(detections)} objects:")
        
        for det in detections:
            print(f"   • {det['class']}: {det['confidence']:.2%} at {det['bbox']}")
        
        return detections
    
    def semantic_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Perform semantic segmentation on NPU"""
        print("\n🎨 Semantic Segmentation on NPU")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # NPU inference
        start = time.perf_counter()
        segmentation_mask = self.segment_npu(processed)
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"⚡ Inference time: {inference_time:.2f}ms")
        print(f"🖼️ Segmentation shape: {segmentation_mask.shape}")
        
        # Count segments
        unique_segments = np.unique(segmentation_mask)
        print(f"📊 Found {len(unique_segments)} segments")
        
        return segmentation_mask
    
    def face_recognition(self, image: np.ndarray) -> list:
        """Perform face recognition on NPU"""
        print("\n👤 Face Recognition on NPU")
        
        # Detect faces
        start = time.perf_counter()
        faces = self.detect_faces_npu(image)
        
        # Extract face embeddings
        embeddings = []
        for face in faces:
            embedding = self.extract_face_embedding_npu(face)
            embeddings.append(embedding)
        
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"⚡ Inference time: {inference_time:.2f}ms")
        print(f"👥 Detected {len(faces)} faces")
        
        return embeddings
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for NPU"""
        # Resize to 224x224 (standard for many models)
        if image.shape[:2] != (224, 224):
            # Simple resize (in practice, use proper interpolation)
            image = image[:224, :224] if image.shape[0] >= 224 else image
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        return image
    
    def extract_features_npu(self, image: np.ndarray) -> np.ndarray:
        """Extract features using NPU"""
        # Simulate NPU feature extraction
        batch_size = image.shape[0]
        features = np.random.randn(batch_size, 2048).astype(np.float32)
        return features
    
    def classify_npu(self, features: np.ndarray) -> np.ndarray:
        """Classify using NPU"""
        # Simulate NPU classification
        num_classes = 1000  # ImageNet classes
        logits = np.random.randn(features.shape[0], num_classes).astype(np.float32)
        return logits
    
    def detect_objects_npu(self, image: np.ndarray) -> list:
        """Detect objects using NPU"""
        # Simulate NPU object detection
        detections = []
        num_objects = np.random.randint(1, 6)
        
        classes = ['person', 'car', 'dog', 'cat', 'bicycle', 'tree', 'building']
        
        for _ in range(num_objects):
            det = {
                'class': np.random.choice(classes),
                'confidence': np.random.uniform(0.7, 0.99),
                'bbox': [
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(20, 100),
                    np.random.randint(20, 100)
                ]
            }
            detections.append(det)
        
        return detections
    
    def segment_npu(self, image: np.ndarray) -> np.ndarray:
        """Semantic segmentation on NPU"""
        # Simulate NPU segmentation
        height, width = 224, 224
        num_classes = 21  # PASCAL VOC classes
        
        # Generate random segmentation mask
        segmentation = np.random.randint(0, num_classes, (height, width))
        return segmentation.astype(np.uint8)
    
    def detect_faces_npu(self, image: np.ndarray) -> list:
        """Detect faces using NPU"""
        # Simulate face detection
        num_faces = np.random.randint(0, 4)
        faces = []
        
        for _ in range(num_faces):
            # Random face crop
            x = np.random.randint(0, image.shape[1] - 50)
            y = np.random.randint(0, image.shape[0] - 50)
            w = np.random.randint(30, 80)
            h = np.random.randint(30, 80)
            
            face_crop = image[y:y+h, x:x+w] if len(image.shape) > 2 else None
            faces.append(face_crop)
        
        return faces
    
    def extract_face_embedding_npu(self, face: np.ndarray) -> np.ndarray:
        """Extract face embedding using NPU"""
        # Simulate NPU face embedding
        embedding_size = 512
        embedding = np.random.randn(embedding_size).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def get_top_k_predictions(self, logits: np.ndarray, k: int = 5) -> list:
        """Get top-k predictions"""
        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get top-k
        top_k_idx = np.argsort(probs[0])[-k:][::-1]
        
        # Mock class names
        class_names = [f"class_{i}" for i in range(len(probs[0]))]
        
        predictions = []
        for idx in top_k_idx:
            predictions.append((class_names[idx], probs[0][idx]))
        
        return predictions
    
    def benchmark_vision_tasks(self):
        """Benchmark various vision tasks"""
        print("\n📊 Benchmarking Computer Vision Tasks on NPU")
        print("=" * 50)
        
        # Create dummy image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        tasks = [
            ("Image Classification", lambda: self.image_classification(image)),
            ("Object Detection", lambda: self.object_detection(image)),
            ("Semantic Segmentation", lambda: self.semantic_segmentation(image)),
            ("Face Recognition", lambda: self.face_recognition(image))
        ]
        
        results = []
        
        for task_name, task_func in tasks:
            print(f"\n🔄 Benchmarking: {task_name}")
            
            # Warmup
            for _ in range(3):
                task_func()
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.perf_counter()
                task_func()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results.append({
                'task': task_name,
                'avg_ms': avg_time,
                'std_ms': std_time,
                'fps': 1000 / avg_time
            })
            
            print(f"   Average: {avg_time:.2f}ms (±{std_time:.2f}ms)")
            print(f"   FPS: {1000/avg_time:.1f}")
        
        # Summary
        print("\n📈 Benchmark Summary:")
        print("-" * 40)
        for result in results:
            print(f"{result['task']}: {result['avg_ms']:.2f}ms ({result['fps']:.1f} FPS)")

def main():
    """Main demo"""
    print("🐉 DragonNPU Computer Vision Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = NPUVisionProcessor()
    
    if not processor.initialize():
        print("Failed to initialize NPU")
        return
    
    # Create dummy image
    print("\n📸 Creating test image...")
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Run demos
    print("\n🎯 Running Computer Vision Tasks:")
    
    # 1. Image Classification
    processor.image_classification(image)
    
    # 2. Object Detection
    processor.object_detection(image)
    
    # 3. Semantic Segmentation
    processor.semantic_segmentation(image)
    
    # 4. Face Recognition
    processor.face_recognition(image)
    
    # Run benchmark
    print("\n" + "="*50)
    processor.benchmark_vision_tasks()
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()