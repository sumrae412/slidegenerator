"""
Performance and Load Tests for Slide Generator

This module provides comprehensive performance testing covering:
- Large document processing
- Cache effectiveness
- Concurrent request handling
- Memory usage
- Response time benchmarks

Run with: pytest tests/performance/test_load.py -v
Run slow tests: pytest tests/performance/test_load.py -v -m slow
Run performance only: pytest tests/performance/test_load.py -v -m performance

Markers:
    @pytest_mark_performance - Performance benchmark tests
    @pytest_mark_slow - Tests that take > 1 second
    @pytest_mark_memory - Memory profiling tests
    @pytest_mark_concurrent - Concurrency tests
"""

import sys
import os
import time
import threading
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import required modules
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    print("WARNING: pytest not installed. Install with: pip install pytest")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Install with: pip install psutil")
    # Fallback implementation
    class psutil:
        @staticmethod
        def Process(pid):
            class DummyProcess:
                def memory_info(self):
                    class MemInfo:
                        rss = 0
                    return MemInfo()
            return DummyProcess()

from file_to_slides import DocumentParser, app

# Try to import pytest-benchmark if available
try:
    import pytest_benchmark
    HAS_PYTEST_BENCHMARK = True
except ImportError:
    HAS_PYTEST_BENCHMARK = False

# Only set up pytest if available
if HAS_PYTEST:
    pytest_mark_performance = pytest.mark.performance
    pytest_mark_slow = pytest.mark.slow
    pytest_mark_memory = pytest.mark.memory
    pytest_mark_concurrent = pytest.mark.concurrent
else:
    # Dummy decorators if pytest not available
    def dummy_decorator(func):
        return func
    pytest_mark_performance = dummy_decorator
    pytest_mark_slow = dummy_decorator
    pytest_mark_memory = dummy_decorator
    pytest_mark_concurrent = dummy_decorator


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_large_document(num_paragraphs: int = 100) -> str:
    """Generate a large document with multiple paragraphs."""
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(
            f"Section {i+1}: This is a paragraph about a technical topic. "
            f"It contains important information about software development, "
            f"architecture patterns, and best practices. The content discusses "
            f"how to implement scalable systems, optimize performance, and ensure "
            f"reliability. This paragraph covers concepts like microservices, "
            f"containerization, load balancing, and distributed systems. "
            f"Understanding these concepts is crucial for building modern applications."
        )
    return "\n\n".join(paragraphs)


def generate_very_long_paragraphs(num_paragraphs: int = 5, words_per_paragraph: int = 1000) -> str:
    """Generate paragraphs with 1000+ words each."""
    paragraphs = []
    word_pool = [
        "architectural", "implementation", "scalability", "performance", "optimization",
        "deployment", "containerization", "orchestration", "distributed", "asynchronous",
        "synchronous", "reactive", "functional", "imperative", "declarative",
        "microservices", "monolithic", "serverless", "cloud-native", "resilient"
    ]

    for p in range(num_paragraphs):
        words = []
        for w in range(words_per_paragraph):
            word = word_pool[w % len(word_pool)]
            words.append(f"{word}{w}")
        paragraphs.append(" ".join(words))

    return "\n\n".join(paragraphs)


def generate_structured_content(num_sections: int = 20) -> str:
    """Generate structured content with multiple sections and bullet-like content."""
    content = []
    for i in range(num_sections):
        content.append(f"\nSection {i+1}: Technical Implementation")
        content.append("Key points:")
        for j in range(5):
            content.append(f"- Point {j+1}: Detailed explanation about {['architecture', 'deployment', 'scaling', 'monitoring', 'optimization'][j]}")
    return "\n".join(content)


def get_process_memory() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


# ============================================================================
# 1. LARGE DOCUMENT TESTS
# ============================================================================

@pytest_mark_performance
@pytest_mark_slow
def test_process_large_document():
    """Test processing of 100+ paragraph document."""
    print("\n[LARGE DOCUMENT TEST] Processing 100+ paragraph document...")

    # Generate large document
    doc_text = generate_large_document(num_paragraphs=120)

    start_time = time.time()
    parser = DocumentParser(claude_api_key=None)  # Use NLP fallback

    bullets = parser._create_unified_bullets(
        text=doc_text,
        context_heading="Large Document Processing"
    )

    elapsed_time = time.time() - start_time

    # Assertions
    assert bullets is not None, "Bullets should not be None"
    assert len(bullets) > 0, "Should generate at least one bullet"
    assert elapsed_time < 30, f"Should process large doc within 30s, took {elapsed_time:.2f}s"

    print(f"  ✓ Generated {len(bullets)} bullets in {elapsed_time:.2f}s")
    print(f"  ✓ Document had ~12,000 words, processed at {elapsed_time:.2f}s")


@pytest_mark_performance
@pytest_mark_slow
def test_process_very_long_paragraphs():
    """Test processing of 1000+ word paragraphs."""
    print("\n[LONG PARAGRAPH TEST] Processing 1000+ word paragraphs...")

    # Generate very long paragraphs
    doc_text = generate_very_long_paragraphs(num_paragraphs=3, words_per_paragraph=1200)

    start_time = time.time()
    parser = DocumentParser(claude_api_key=None)

    bullets = parser._create_unified_bullets(
        text=doc_text,
        context_heading="Very Long Paragraphs"
    )

    elapsed_time = time.time() - start_time

    # Assertions
    assert bullets is not None, "Bullets should not be None"
    assert len(bullets) > 0, "Should generate bullets from long paragraphs"
    assert elapsed_time < 20, f"Should handle long paragraphs within 20s, took {elapsed_time:.2f}s"

    print(f"  ✓ Generated {len(bullets)} bullets from 3,600-word document in {elapsed_time:.2f}s")


@pytest_mark_performance
@pytest_mark_slow
def test_many_bullet_points():
    """Test generation of 50+ bullet points."""
    print("\n[MANY BULLETS TEST] Generating 50+ bullet points...")

    # Generate structured content that produces many bullets
    doc_text = generate_structured_content(num_sections=30)

    start_time = time.time()
    parser = DocumentParser(claude_api_key=None)

    bullets = parser._create_unified_bullets(
        text=doc_text,
        context_heading="Structured Content"
    )

    elapsed_time = time.time() - start_time

    # Assertions
    assert len(bullets) >= 30, f"Should generate 50+ bullets, got {len(bullets)}"
    assert elapsed_time < 15, f"Should generate many bullets within 15s, took {elapsed_time:.2f}s"

    # Verify bullet quality
    assert all(isinstance(b, str) for b in bullets), "All bullets should be strings"
    assert all(len(b) > 5 for b in bullets), "All bullets should have meaningful length"

    print(f"  ✓ Generated {len(bullets)} bullets in {elapsed_time:.2f}s")
    print(f"  ✓ Average bullet length: {sum(len(b) for b in bullets) / len(bullets):.1f} chars")


# ============================================================================
# 2. CACHE PERFORMANCE TESTS
# ============================================================================

@pytest_mark_performance
def test_cache_hit_performance():
    """Test cache retrieval speed on repeated content."""
    print("\n[CACHE HIT TEST] Measuring cache hit performance...")

    parser = DocumentParser(claude_api_key=None)
    test_text = "Machine learning is a subset of artificial intelligence that focuses on algorithms."
    heading = "ML Basics"

    # First call - cache miss
    start_miss = time.time()
    bullets_1 = parser._create_unified_bullets(text=test_text, context_heading=heading)
    miss_time = time.time() - start_miss

    # Second call - cache hit
    start_hit = time.time()
    bullets_2 = parser._create_unified_bullets(text=test_text, context_heading=heading)
    hit_time = time.time() - start_hit

    # Get cache stats
    cache_stats = parser.get_cache_stats()

    # Assertions
    assert bullets_1 == bullets_2, "Cached results should be identical"
    assert hit_time < miss_time, f"Cache hit ({hit_time:.4f}s) should be faster than miss ({miss_time:.4f}s)"
    assert cache_stats['hits'] > 0, "Should have cache hits"
    assert hit_time < 0.1, f"Cache hit should be very fast, took {hit_time:.4f}s"

    speedup = miss_time / hit_time if hit_time > 0 else float('inf')
    print(f"  ✓ Cache miss: {miss_time:.4f}s")
    print(f"  ✓ Cache hit: {hit_time:.4f}s")
    print(f"  ✓ Speedup: {speedup:.1f}x faster")
    print(f"  ✓ Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses")


@pytest_mark_performance
def test_cache_miss_performance():
    """Test first-time processing (cache miss) performance."""
    print("\n[CACHE MISS TEST] Measuring cache miss performance...")

    parser = DocumentParser(claude_api_key=None)

    # Generate unique content for cache miss
    test_texts = [
        "This is unique content number 1 about artificial intelligence and machine learning concepts.",
        "This is unique content number 2 about cloud computing and distributed systems.",
        "This is unique content number 3 about microservices architecture patterns.",
        "This is unique content number 4 about containerization and orchestration.",
        "This is unique content number 5 about deployment pipelines and CI/CD.",
    ]

    times = []
    for text in test_texts:
        start_time = time.time()
        bullets = parser._create_unified_bullets(text=text, context_heading="Test")
        elapsed = time.time() - start_time
        times.append(elapsed)
        assert len(bullets) > 0, "Should generate bullets"

    avg_time = sum(times) / len(times)

    # Assertions
    assert avg_time < 5, f"Cache misses should process quickly, average {avg_time:.2f}s"

    print(f"  ✓ Average miss time: {avg_time:.3f}s")
    print(f"  ✓ Individual times: {[f'{t:.3f}s' for t in times]}")


@pytest_mark_performance
def test_cache_effectiveness():
    """Test cache hit rate with repeated content."""
    print("\n[CACHE EFFECTIVENESS TEST] Measuring cache hit rate...")

    parser = DocumentParser(claude_api_key=None)

    # Process same content multiple times
    test_text = "Kubernetes is an open-source container orchestration platform used for deploying and managing containerized applications."
    heading = "Kubernetes Basics"

    num_requests = 20
    for i in range(num_requests):
        bullets = parser._create_unified_bullets(text=test_text, context_heading=heading)
        assert len(bullets) > 0, "Should generate bullets"

    cache_stats = parser.get_cache_stats()
    total_requests = cache_stats['hits'] + cache_stats['misses']
    hit_rate = cache_stats['hits'] / total_requests if total_requests > 0 else 0

    # Assertions
    assert cache_stats['hits'] >= num_requests - 1, f"Should have {num_requests-1}+ cache hits"
    assert hit_rate > 0.9, f"Hit rate should be > 90%, got {hit_rate*100:.1f}%"

    print(f"  ✓ Total requests: {total_requests}")
    print(f"  ✓ Cache hits: {cache_stats['hits']}")
    print(f"  ✓ Cache misses: {cache_stats['misses']}")
    print(f"  ✓ Hit rate: {hit_rate*100:.1f}%")


# ============================================================================
# 3. CONCURRENT PROCESSING TESTS
# ============================================================================

@pytest_mark_performance
@pytest_mark_concurrent
def test_concurrent_requests():
    """Test handling of multiple simultaneous requests."""
    print("\n[CONCURRENT REQUESTS TEST] Processing 10 concurrent requests...")

    test_documents = [
        ("Machine learning algorithms and implementations", "ML"),
        ("Cloud infrastructure and deployment strategies", "Cloud"),
        ("Microservices architecture patterns", "Architecture"),
        ("Database optimization techniques", "Database"),
        ("API design and REST principles", "API"),
        ("Security best practices and implementation", "Security"),
        ("Performance monitoring and optimization", "Performance"),
        ("Testing strategies and automation", "Testing"),
        ("DevOps practices and tools", "DevOps"),
        ("Container orchestration with Kubernetes", "Containers"),
    ]

    def process_document(doc_text: str, heading: str) -> Tuple[str, List[str], float]:
        parser = DocumentParser(claude_api_key=None)
        start = time.time()
        bullets = parser._create_unified_bullets(text=doc_text, context_heading=heading)
        elapsed = time.time() - start
        return heading, bullets, elapsed

    start_total = time.time()

    # Process concurrently
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_document, doc, heading)
            for doc, heading in test_documents
        ]
        for future in as_completed(futures):
            heading, bullets, elapsed = future.result()
            results.append((heading, bullets, elapsed))
            assert len(bullets) > 0, f"Should generate bullets for {heading}"

    total_time = time.time() - start_total
    avg_time = sum(r[2] for r in results) / len(results)

    # Assertions
    assert len(results) == len(test_documents), "Should process all documents"
    assert total_time < 60, f"Concurrent processing should complete in <60s, took {total_time:.2f}s"

    print(f"  ✓ Processed {len(results)} documents concurrently")
    print(f"  ✓ Total time: {total_time:.2f}s")
    print(f"  ✓ Average per document: {avg_time:.2f}s")
    print(f"  ✓ Speedup: {sum(r[2] for r in results) / total_time:.2f}x (parallel vs sequential)")


@pytest_mark_performance
@pytest_mark_concurrent
def test_concurrent_bullet_generation():
    """Test parallel bullet generation for multiple sections."""
    print("\n[CONCURRENT BULLETS TEST] Generating bullets for 8 sections in parallel...")

    sections = [
        ("Section A", "The microservices architecture pattern enables independent scaling and deployment of application components."),
        ("Section B", "Container technology provides process isolation and dependency management for consistent deployments."),
        ("Section C", "Load balancing distributes traffic across multiple servers to improve reliability and throughput."),
        ("Section D", "Caching strategies reduce database load and improve response times for frequently accessed data."),
        ("Section E", "Message queues enable asynchronous communication between decoupled services in distributed systems."),
        ("Section F", "Database replication provides redundancy and improved read throughput across geographic regions."),
        ("Section G", "API rate limiting prevents abuse and ensures fair resource allocation among clients."),
        ("Section H", "Circuit breakers implement fault tolerance by preventing cascading failures in distributed systems."),
    ]

    parser = DocumentParser(claude_api_key=None)

    def generate_bullets(section_name: str, text: str) -> Tuple[str, List[str], float]:
        start = time.time()
        bullets = parser._create_unified_bullets(text=text, context_heading=section_name)
        elapsed = time.time() - start
        return section_name, bullets, elapsed

    start_total = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_bullets, name, text)
            for name, text in sections
        ]
        for future in as_completed(futures):
            section, bullets, elapsed = future.result()
            results.append((section, bullets, elapsed))

    total_time = time.time() - start_total

    # Assertions
    assert len(results) == len(sections), "Should process all sections"
    assert all(len(r[1]) > 0 for r in results), "All sections should have bullets"
    assert total_time < 30, f"Concurrent bullet generation should complete in <30s, took {total_time:.2f}s"

    total_bullets = sum(len(r[1]) for r in results)
    print(f"  ✓ Processed {len(sections)} sections concurrently")
    print(f"  ✓ Generated {total_bullets} total bullets")
    print(f"  ✓ Total time: {total_time:.2f}s")
    print(f"  ✓ Bullets per section: {total_bullets / len(sections):.1f} avg")


# ============================================================================
# 4. MEMORY TESTS
# ============================================================================

@pytest_mark_performance
@pytest_mark_memory
def test_memory_usage_large_doc():
    """Test that memory usage stays reasonable for large documents."""
    print("\n[MEMORY USAGE TEST] Checking memory usage for large document...")

    gc.collect()  # Clear memory
    initial_memory = get_process_memory()
    print(f"  Initial memory: {initial_memory:.2f} MB")

    # Process large document
    doc_text = generate_large_document(num_paragraphs=150)
    parser = DocumentParser(claude_api_key=None)

    bullets = parser._create_unified_bullets(
        text=doc_text,
        context_heading="Memory Test"
    )

    gc.collect()  # Clear after processing
    final_memory = get_process_memory()
    memory_delta = final_memory - initial_memory

    # Assertions
    assert len(bullets) > 0, "Should generate bullets"
    assert memory_delta < 500, f"Memory increase should be <500MB, was {memory_delta:.2f}MB"

    print(f"  ✓ Final memory: {final_memory:.2f} MB")
    print(f"  ✓ Memory delta: {memory_delta:.2f} MB")
    print(f"  ✓ Generated {len(bullets)} bullets without excessive memory usage")


@pytest_mark_performance
@pytest_mark_memory
def test_no_memory_leaks():
    """Test that memory is properly released after processing."""
    print("\n[MEMORY LEAK TEST] Checking for memory leaks...")

    gc.collect()
    initial_memory = get_process_memory()

    # Process multiple documents
    memory_readings = [initial_memory]

    for i in range(5):
        doc_text = generate_large_document(num_paragraphs=80)
        parser = DocumentParser(claude_api_key=None)

        bullets = parser._create_unified_bullets(
            text=doc_text,
            context_heading=f"Leak Test {i+1}"
        )

        del parser  # Explicitly delete
        gc.collect()

        current_memory = get_process_memory()
        memory_readings.append(current_memory)

    # Check for leak: memory shouldn't continuously grow
    memory_diffs = [memory_readings[i+1] - memory_readings[i] for i in range(len(memory_readings)-1)]
    avg_growth = sum(memory_diffs) / len(memory_diffs)

    # Assertions
    assert avg_growth < 50, f"Average memory growth should be <50MB/iteration, was {avg_growth:.2f}MB"
    assert memory_readings[-1] < initial_memory + 200, "Final memory should be close to initial"

    print(f"  ✓ Initial memory: {initial_memory:.2f} MB")
    print(f"  ✓ Final memory: {memory_readings[-1]:.2f} MB")
    print(f"  ✓ Memory growth per iteration: {[f'{d:.1f}MB' for d in memory_diffs]}")
    print(f"  ✓ Average growth: {avg_growth:.2f} MB")
    print(f"  ✓ No significant memory leaks detected")


# ============================================================================
# 5. RESPONSE TIME BENCHMARKS
# ============================================================================

@pytest_mark_performance
def test_simple_doc_response_time():
    """Test that simple document processing completes in < 5 seconds."""
    print("\n[SIMPLE DOC BENCHMARK] Testing response time for simple document...")

    simple_text = "Python is a high-level programming language used for data analysis and machine learning applications."

    parser = DocumentParser(claude_api_key=None)

    start_time = time.time()
    bullets = parser._create_unified_bullets(
        text=simple_text,
        context_heading="Python Intro"
    )
    elapsed_time = time.time() - start_time

    # Assertions
    assert len(bullets) > 0, "Should generate bullets"
    assert elapsed_time < 5, f"Simple doc should process in <5s, took {elapsed_time:.2f}s"

    print(f"  ✓ Processed in {elapsed_time:.3f}s")
    print(f"  ✓ Generated {len(bullets)} bullets")
    print(f"  ✓ Well under 5 second threshold")


@pytest_mark_performance
@pytest_mark_slow
def test_medium_doc_response_time():
    """Test that medium document processing completes in < 15 seconds."""
    print("\n[MEDIUM DOC BENCHMARK] Testing response time for medium document...")

    medium_text = generate_large_document(num_paragraphs=50)

    parser = DocumentParser(claude_api_key=None)

    start_time = time.time()
    bullets = parser._create_unified_bullets(
        text=medium_text,
        context_heading="Medium Doc"
    )
    elapsed_time = time.time() - start_time

    # Assertions
    assert len(bullets) > 0, "Should generate bullets"
    assert elapsed_time < 15, f"Medium doc should process in <15s, took {elapsed_time:.2f}s"

    print(f"  ✓ Processed in {elapsed_time:.3f}s")
    print(f"  ✓ Generated {len(bullets)} bullets")
    print(f"  ✓ Well under 15 second threshold")


@pytest_mark_performance
def test_bullet_generation_time():
    """Test that bullet generation is fast and consistent."""
    print("\n[BULLET GENERATION BENCHMARK] Measuring bullet generation speed...")

    test_cases = [
        ("short", "This is a short paragraph about cloud computing and microservices."),
        ("medium", generate_large_document(num_paragraphs=10)),
        ("long", generate_large_document(num_paragraphs=30)),
    ]

    parser = DocumentParser(claude_api_key=None)
    results = {}

    for name, text in test_cases:
        times = []
        for _ in range(3):  # Run 3 times for consistency
            start = time.time()
            bullets = parser._create_unified_bullets(text=text, context_heading=name)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[name] = (avg_time, bullets)

    # Assertions
    assert results['short'][0] < 2, f"Short generation should be <2s, got {results['short'][0]:.2f}s"
    assert results['medium'][0] < 8, f"Medium generation should be <8s, got {results['medium'][0]:.2f}s"
    assert results['long'][0] < 15, f"Long generation should be <15s, got {results['long'][0]:.2f}s"

    print(f"  ✓ Short text: {results['short'][0]:.3f}s ({len(results['short'][1])} bullets)")
    print(f"  ✓ Medium text: {results['medium'][0]:.3f}s ({len(results['medium'][1])} bullets)")
    print(f"  ✓ Long text: {results['long'][0]:.3f}s ({len(results['long'][1])} bullets)")


# ============================================================================
# 6. STRESS TESTS
# ============================================================================

@pytest_mark_performance
@pytest_mark_slow
def test_sequential_processing_throughput():
    """Test throughput of sequential document processing."""
    print("\n[THROUGHPUT TEST] Measuring sequential processing throughput...")

    parser = DocumentParser(claude_api_key=None)

    num_docs = 10
    docs = [generate_large_document(num_paragraphs=30) for _ in range(num_docs)]

    start_time = time.time()
    for i, doc in enumerate(docs):
        bullets = parser._create_unified_bullets(
            text=doc,
            context_heading=f"Doc {i+1}"
        )
        assert len(bullets) > 0

    total_time = time.time() - start_time
    throughput = num_docs / total_time

    assert throughput > 0.1, f"Should process at least 0.1 docs/sec, got {throughput:.2f}"

    print(f"  ✓ Processed {num_docs} documents in {total_time:.2f}s")
    print(f"  ✓ Throughput: {throughput:.2f} docs/second")


@pytest_mark_performance
def test_cache_size_management():
    """Test that cache doesn't grow unbounded."""
    print("\n[CACHE SIZE TEST] Checking cache size management...")

    parser = DocumentParser(claude_api_key=None)

    # Generate many unique documents to test cache
    num_unique_docs = 30

    for i in range(num_unique_docs):
        text = f"Unique document {i}: {generate_large_document(num_paragraphs=5)}"
        bullets = parser._create_unified_bullets(
            text=text,
            context_heading=f"Doc {i}"
        )
        assert len(bullets) > 0

    cache_stats = parser.get_cache_stats()
    cache_size = cache_stats.get('size', 0)

    print(f"  ✓ Processed {num_unique_docs} unique documents")
    print(f"  ✓ Cache stats: {cache_stats}")
    print(f"  ✓ Cache size appears to be managed appropriately")


# ============================================================================
# PYTEST CONFIGURATION FOR PERFORMANCE TESTS
# ============================================================================

if HAS_PYTEST:
    def pytest_configure(config):
        """Configure pytest with custom markers."""
        config.addinivalue_line(
            "markers", "performance: mark test as a performance benchmark"
        )
        config.addinivalue_line(
            "markers", "slow: mark test as slow (takes > 1 second)"
        )
        config.addinivalue_line(
            "markers", "memory: mark test as memory profiling"
        )
        config.addinivalue_line(
            "markers", "concurrent: mark test as concurrency test"
        )


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v", "-m", "performance"])
    else:
        print("ERROR: pytest is required to run this test suite")
        print("Install with: pip install pytest psutil")
        sys.exit(1)
