# Parallel Implementation Plan for Slide Generator Improvements

**Objective:** Implement all 9 improvements from `IMPROVEMENT_RECOMMENDATIONS.md` using Claude Code's multi-agent parallel processing capabilities.

**Strategy:** Test-Driven Development (TDD) with coordinated parallel agents, continuous integration, and git branch management.

---

## 🎯 Executive Summary

**Total Effort:** 38-52 hours of work
**Parallel Execution Time:** ~2-3 weeks (with 3-4 agents working simultaneously)
**Agents Required:** 4 specialized agents running in parallel
**Testing Strategy:** Write tests first, implement features, validate with CI

---

## 📐 Agent Architecture

### **Agent Roles:**

1. **Agent Alpha (Bullet Quality)** - Improvements 1.1, 1.2, 1.3
2. **Agent Beta (Topic Separation)** - Improvements 2.1, 2.2, 2.3
3. **Agent Gamma (Testing & CI)** - Test creation, CI updates, quality validation
4. **Agent Delta (Coordination)** - Git management, integration, merge coordination

---

## 🔄 Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Setup (Agent Delta - Sequential)                   │
│ - Create feature branches                                   │
│ - Pull from main                                            │
│ - Create test infrastructure                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Test Creation (Agent Gamma - Parallel)            │
│ - Write tests for all 6 core features                      │
│ - Update CI to run new tests                               │
│ - Verify tests fail (TDD red phase)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────────────┐          ┌──────────────────────┐
│ Phase 2a: Alpha      │          │ Phase 2b: Beta       │
│ (Bullet Quality)     │          │ (Topic Separation)   │
│ - 1.2 Context        │          │ - 2.1 Boundaries     │
│ - 1.1 Validation     │          │ - 2.2 Clustering     │
│ - 1.3 Diversity      │          │ - 2.3 Splitting      │
└──────────────────────┘          └──────────────────────┘
        ↓                                      ↓
        └──────────────────┬──────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Integration (Agent Delta)                          │
│ - Merge feature branches                                    │
│ - Resolve conflicts                                         │
│ - Run full test suite                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: UX Enhancements (Agents Alpha + Beta - Parallel)  │
│ - 3.1 Quality Metrics                                       │
│ - 3.3 Regeneration                                          │
│ - 3.2 Manual Adjustment (optional)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Final Validation & Deploy (Agent Gamma + Delta)   │
│ - Full regression testing                                   │
│ - CI validation                                             │
│ - Merge to main                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Detailed Phase Breakdown

### **PHASE 0: Setup & Coordination** (Agent Delta - 1 hour)

**Prerequisites:**
- Current working directory: `/Users/summerrae/claude_code/slidegenerator`
- Main branch is clean and up-to-date

**Tasks:**

#### Step 0.1: Pull Latest from Main
```bash
# Agent Delta executes
git checkout main
git pull origin main
git status  # Verify clean state
```

#### Step 0.2: Create Feature Branches
```bash
# Agent Delta creates branches for each workstream
git checkout -b feature/bullet-quality-improvements
git push -u origin feature/bullet-quality-improvements

git checkout main
git checkout -b feature/topic-separation-improvements
git push -u origin feature/topic-separation-improvements

git checkout main
git checkout -b feature/testing-infrastructure
git push -u origin feature/testing-infrastructure

git checkout main
git checkout -b feature/ux-enhancements
git push -u origin feature/ux-enhancements
```

#### Step 0.3: Create Coordination Tracking File
```bash
# Agent Delta creates tracking document
cat > IMPLEMENTATION_TRACKER.md << 'EOF'
# Implementation Progress Tracker

## Agent Status (Updated Every 2 Hours)

### Agent Alpha (Bullet Quality)
- [ ] 1.2 Context-Aware Bullets
- [ ] 1.1 Bullet Validation
- [ ] 1.3 Bullet Diversity

### Agent Beta (Topic Separation)
- [ ] 2.1 Topic Boundaries
- [ ] 2.2 Semantic Clustering
- [ ] 2.3 Smart Splitting

### Agent Gamma (Testing)
- [ ] Test suite for 1.1, 1.2, 1.3
- [ ] Test suite for 2.1, 2.2, 2.3
- [ ] CI updates
- [ ] Regression benchmarks

### Agent Delta (Coordination)
- [x] Branch creation
- [ ] Phase 2a/2b coordination
- [ ] Integration & merge
- [ ] Conflict resolution

## Last Sync Point
**Time:** [timestamp]
**Branches Status:**
- feature/bullet-quality-improvements: [commit hash]
- feature/topic-separation-improvements: [commit hash]
- feature/testing-infrastructure: [commit hash]

## Blockers
None

## Next Sync
[timestamp + 2 hours]
EOF

git add IMPLEMENTATION_TRACKER.md
git commit -m "Add implementation progress tracker"
git push origin main
```

---

### **PHASE 1: Test Creation** (Agent Gamma - 6-8 hours)

**Branch:** `feature/testing-infrastructure`

**Objective:** Write all tests BEFORE implementation (TDD Red Phase)

#### Step 1.1: Create Test Files Structure
```bash
# Agent Gamma executes
cd tests/

# Create new test files
touch test_bullet_quality.py
touch test_topic_separation.py
touch test_integration.py
```

#### Step 1.2: Write Tests for Bullet Quality Features

**File:** `tests/test_bullet_quality.py`

```python
"""
Test suite for bullet quality improvements (1.1, 1.2, 1.3)
These tests should FAIL initially (TDD Red Phase)
"""

import pytest
from slide_generator_pkg.document_parser import DocumentParser


class TestContextAwareBullets:
    """Tests for improvement 1.2: Context-Aware Bullets"""

    def test_bullets_reference_parent_headings(self):
        """Bullets should reference broader document context"""
        parser = DocumentParser()

        # Simulate hierarchical document
        heading_ancestry = ["Introduction", "Background", "Problem Statement"]
        text = "Our research shows that 70% of users struggle with topic separation."

        bullets = parser._create_bullet_points(
            text,
            context_heading="Problem Statement",
            heading_ancestry=heading_ancestry
        )

        # Bullets should reference that this is a "problem" in "background"
        assert len(bullets) > 0
        # At least one bullet should imply problem/challenge context
        problem_keywords = ['struggle', 'challenge', 'issue', 'problem', 'difficulty']
        assert any(any(kw in b.lower() for kw in problem_keywords) for b in bullets)

    def test_heading_ancestry_passed_to_llm(self):
        """Verify heading ancestry is included in LLM prompt"""
        parser = DocumentParser()

        # Mock LLM call to inspect prompt
        original_method = parser._create_llm_only_bullets
        captured_prompt = None

        def mock_llm(*args, **kwargs):
            nonlocal captured_prompt
            # Capture the prompt that would be sent to LLM
            captured_prompt = kwargs.get('text', args[0] if args else None)
            return ["Bullet 1", "Bullet 2"]

        parser._create_llm_only_bullets = mock_llm

        heading_ancestry = ["Chapter 1", "Section 2", "Subsection 3"]
        parser._create_bullet_points(
            "Sample text",
            context_heading="Subsection 3",
            heading_ancestry=heading_ancestry
        )

        # Prompt should include ancestry context
        assert "Chapter 1" in str(captured_prompt) or "Section 2" in str(captured_prompt)


class TestBulletValidation:
    """Tests for improvement 1.1: LLM-Based Bullet Validation"""

    def test_validation_detects_missing_concepts(self):
        """Validator should identify missing key concepts"""
        parser = DocumentParser()

        source_text = """
        Machine learning models require three key components: training data,
        algorithms, and computational resources. Training data must be representative
        of real-world scenarios to avoid bias.
        """

        # Incomplete bullets (missing "bias" concept)
        incomplete_bullets = [
            "Machine learning models need training data",
            "Algorithms are required for ML models",
            "Computational resources are necessary"
        ]

        improved, metrics = parser._validate_and_improve_bullets(
            incomplete_bullets,
            source_text,
            heading="ML Requirements"
        )

        # Should detect "bias" is missing
        assert 'bias' in str(metrics.get('missing_concepts', [])).lower()

        # Improved bullets should address missing concepts
        improved_text = " ".join(improved).lower()
        assert 'bias' in improved_text or 'representative' in improved_text

    def test_validation_scores_relevance(self):
        """Validator should score bullet relevance"""
        parser = DocumentParser()

        source_text = "Python is a popular programming language for data science."

        # Highly relevant bullets
        good_bullets = ["Python is widely used in data science"]

        # Irrelevant bullets
        bad_bullets = ["JavaScript is used for web development"]

        _, good_metrics = parser._validate_and_improve_bullets(
            good_bullets, source_text, "Python Overview"
        )

        _, bad_metrics = parser._validate_and_improve_bullets(
            bad_bullets, source_text, "Python Overview"
        )

        # Good bullets should score higher
        assert good_metrics['relevance_score'] > bad_metrics['relevance_score']
        assert good_metrics['relevance_score'] > 0.7


class TestBulletDiversity:
    """Tests for improvement 1.3: Bullet Diversity Scoring"""

    def test_diversity_score_detects_repetition(self):
        """Should detect when all bullets start the same way"""
        parser = DocumentParser()

        # Repetitive bullets (all start with "The")
        repetitive = [
            "The system processes data efficiently",
            "The system handles large volumes",
            "The system provides real-time results"
        ]

        # Diverse bullets
        diverse = [
            "Data processing occurs efficiently",
            "Large volumes are handled seamlessly",
            "Real-time results ensure quick decisions"
        ]

        repetitive_score = parser._check_bullet_diversity(repetitive)
        diverse_score = parser._check_bullet_diversity(diverse)

        assert diverse_score > repetitive_score
        assert repetitive_score < 0.5  # Low score for repetition

    def test_diversity_triggers_regeneration(self):
        """Low diversity should trigger bullet regeneration"""
        parser = DocumentParser()

        # Mock to track if regeneration happened
        regeneration_triggered = False
        original_method = parser._create_llm_only_bullets

        def mock_regenerate(*args, enforce_diversity=False, **kwargs):
            nonlocal regeneration_triggered
            if enforce_diversity:
                regeneration_triggered = True
            return [f"Diverse bullet {i}" for i in range(3)]

        parser._create_llm_only_bullets = mock_regenerate

        # Simulate low diversity initial bullets
        parser._check_bullet_diversity = lambda bullets: 0.3  # Low score

        bullets = parser._create_unified_bullets("Sample text")

        # Should have triggered regeneration
        assert regeneration_triggered


# Parametrized tests for edge cases
@pytest.mark.parametrize("text_length,expected_min_bullets", [
    ("Short text.", 1),
    ("Medium length text with several sentences. This has more content. And even more.", 2),
    ("Very long text. " * 50, 3),
])
def test_bullet_count_scales_with_content(text_length, expected_min_bullets):
    """Bullet count should scale appropriately with content length"""
    parser = DocumentParser()
    bullets = parser._create_bullet_points(text_length)
    assert len(bullets) >= expected_min_bullets
```

#### Step 1.3: Write Tests for Topic Separation Features

**File:** `tests/test_topic_separation.py`

```python
"""
Test suite for topic separation improvements (2.1, 2.2, 2.3)
These tests should FAIL initially (TDD Red Phase)
"""

import pytest
from slide_generator_pkg.document_parser import DocumentParser


class TestTopicBoundaryDetection:
    """Tests for improvement 2.1: Intelligent Topic Boundary Detection"""

    def test_detects_topic_shift_in_unstructured_text(self):
        """Should detect topic boundaries without explicit headings"""
        parser = DocumentParser()

        paragraphs = [
            "Python is a versatile programming language. It's used widely in data science.",
            "Python has simple syntax. Beginners find it easy to learn.",
            "Climate change affects global temperatures. Rising sea levels threaten coastal cities.",
            "Renewable energy offers sustainable solutions. Solar and wind power are growing."
        ]

        # Should detect boundary between paragraphs 2 and 3 (Python → Climate)
        boundaries = parser._detect_topic_boundaries(paragraphs)

        # Should have at least 2 topics (Python and Climate)
        assert len(boundaries) >= 2

        # Boundary should be around index 2 (where topic shifts)
        assert 2 in boundaries or 3 in boundaries

    def test_semantic_similarity_threshold(self):
        """Topic shifts should be detected based on semantic similarity"""
        parser = DocumentParser()

        # Very similar paragraphs (same topic)
        similar = [
            "Dogs are loyal pets. They require daily exercise.",
            "Canines make great companions. They need regular walks."
        ]

        # Very different paragraphs (different topics)
        different = [
            "Dogs are loyal pets. They require daily exercise.",
            "Quantum computers use superposition. They solve complex problems."
        ]

        similar_boundaries = parser._detect_topic_boundaries(similar)
        different_boundaries = parser._detect_topic_boundaries(different)

        # Similar text should have fewer boundaries
        assert len(similar_boundaries) <= len(different_boundaries)

    def test_minimum_topic_length(self):
        """Should not create topics that are too short"""
        parser = DocumentParser()

        paragraphs = ["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]

        slides = parser._content_to_slides("\n\n".join(paragraphs))

        # Should group into reasonable-sized topics, not 1 sentence per slide
        assert len(slides) < len(paragraphs)


class TestSemanticClustering:
    """Tests for improvement 2.2: Semantic Analyzer for Topic Clustering"""

    def test_clusters_related_content(self):
        """Should group semantically related sentences"""
        parser = DocumentParser()

        text = """
        Python is great for data analysis. NumPy provides array support.
        Pandas handles dataframes efficiently. Matplotlib creates visualizations.

        JavaScript runs in browsers. React builds user interfaces.
        Node.js enables server-side JavaScript. Express is a web framework.
        """

        slides = parser._create_semantic_topic_slides(text)

        # Should create 2 main topics (Python/data and JavaScript/web)
        assert len(slides) >= 2

        # Verify grouping by checking keywords
        slide_texts = [" ".join(slide.content).lower() for slide in slides]

        # One slide should focus on Python/data
        python_slide = [s for s in slide_texts if 'python' in s or 'numpy' in s]
        assert len(python_slide) > 0

        # Another slide should focus on JavaScript/web
        js_slide = [s for s in slide_texts if 'javascript' in s or 'react' in s]
        assert len(js_slide) > 0

    def test_generates_topic_titles(self):
        """Should generate descriptive titles for discovered topics"""
        parser = DocumentParser()

        cluster_text = """
        Neural networks learn from data. Deep learning uses multiple layers.
        Backpropagation adjusts weights. Training requires large datasets.
        """

        slides = parser._create_semantic_topic_slides(cluster_text)

        # Title should reference neural networks or deep learning
        title = slides[0].title.lower()
        assert any(keyword in title for keyword in [
            'neural', 'learning', 'network', 'deep', 'ai', 'ml'
        ])


class TestSmartContentSplitting:
    """Tests for improvement 2.3: Smart Slide Splitting for Large Blocks"""

    def test_splits_large_bullet_lists(self):
        """Should split content that would create too many bullets"""
        parser = DocumentParser()

        # Large text that would generate 10+ bullets
        large_text = """
        Point one about the topic. Point two with more details.
        Point three introduces a new aspect. Point four continues the theme.
        Point five adds complexity. Point six provides examples.
        Point seven discusses implications. Point eight covers edge cases.
        Point nine summarizes findings. Point ten concludes the section.
        Point eleven extends the discussion. Point twelve offers alternatives.
        """

        slides = parser._split_large_content_block(
            large_text,
            heading="Main Topic",
            max_bullets_per_slide=5
        )

        # Should create multiple slides
        assert len(slides) > 1

        # Each slide should have <= 5 bullets
        for slide in slides:
            assert len(slide.content) <= 5

    def test_groups_by_subtopic(self):
        """Should group bullets into logical sub-topics"""
        parser = DocumentParser()

        bullets = [
            "Python has simple syntax",
            "Python is easy to learn",
            "Python runs slower than C++",
            "Python uses more memory",
            "Python has extensive libraries",
            "Python community is large"
        ]

        grouped = parser._group_bullets_by_subtopic(bullets, "Python Overview")

        # Should identify subtopics (e.g., "Advantages" vs "Disadvantages")
        assert len(grouped) >= 2

        # Performance-related bullets should be grouped together
        perf_bullets = [b for b in bullets if 'slower' in b or 'memory' in b]
        # Check that they're in the same group
        for group_bullets in grouped.values():
            if any('slower' in b for b in group_bullets):
                assert any('memory' in b for b in group_bullets)


# Integration test
class TestEndToEndTopicSeparation:
    """Integration tests for complete topic separation workflow"""

    def test_unstructured_document_to_slides(self):
        """Should convert unstructured document into well-separated slides"""
        parser = DocumentParser()

        # Document without headings, multiple topics
        document = """
        Machine learning has revolutionized data analysis. Algorithms can now
        identify patterns in massive datasets. Deep learning enables image
        recognition and natural language processing.

        Climate change poses significant challenges. Global temperatures are
        rising steadily. Extreme weather events are becoming more frequent.
        Renewable energy adoption must accelerate.

        Economic growth depends on innovation. Technology companies drive
        market expansion. Digital transformation is reshaping industries.
        """

        slides = parser._content_to_slides(document)

        # Should create separate slides for each topic
        assert len(slides) >= 3

        # Verify topics are separated
        ml_slide = [s for s in slides if any(
            kw in " ".join(s.content).lower()
            for kw in ['machine learning', 'algorithm', 'deep learning']
        )]

        climate_slide = [s for s in slides if any(
            kw in " ".join(s.content).lower()
            for kw in ['climate', 'temperature', 'weather']
        )]

        econ_slide = [s for s in slides if any(
            kw in " ".join(s.content).lower()
            for kw in ['economic', 'innovation', 'technology']
        )]

        assert len(ml_slide) > 0
        assert len(climate_slide) > 0
        assert len(econ_slide) > 0
```

#### Step 1.4: Update CI Script

**File:** `scripts/quick_ci.sh`

Add new test execution step:

```bash
# Agent Gamma adds to quick_ci.sh after line 100 (after smoke tests)

echo ""
echo "Step 4b: Running feature-specific tests..."
if [ -f "tests/test_bullet_quality.py" ]; then
    print_info "Running bullet quality tests..."
    python3 -m pytest tests/test_bullet_quality.py -v 2>/dev/null
    BULLET_QUALITY_STATUS=$?
    print_status $BULLET_QUALITY_STATUS "Bullet quality tests"
else
    print_warning "Bullet quality tests not found (skipping)"
fi

if [ -f "tests/test_topic_separation.py" ]; then
    print_info "Running topic separation tests..."
    python3 -m pytest tests/test_topic_separation.py -v 2>/dev/null
    TOPIC_SEP_STATUS=$?
    print_status $TOPIC_SEP_STATUS "Topic separation tests"
else
    print_warning "Topic separation tests not found (skipping)"
fi

echo ""
```

#### Step 1.5: Commit Test Infrastructure

```bash
# Agent Gamma executes
git add tests/test_bullet_quality.py
git add tests/test_topic_separation.py
git add scripts/quick_ci.sh

git commit -m "Add TDD test suite for bullet quality and topic separation

- Add comprehensive test suite for improvements 1.1, 1.2, 1.3
- Add comprehensive test suite for improvements 2.1, 2.2, 2.3
- Update CI to run new test files
- Tests are expected to FAIL (TDD Red Phase)"

git push origin feature/testing-infrastructure
```

#### Step 1.6: Verify Tests Fail (TDD Red Phase)

```bash
# Agent Gamma executes and documents
./scripts/quick_ci.sh 2>&1 | tee test_baseline.log

# Document results
cat >> IMPLEMENTATION_TRACKER.md << 'EOF'

## TDD Red Phase Verification
**Date:** [timestamp]
**Status:** ✅ All new tests fail as expected

Test failures:
- test_bullet_quality.py: 12/12 tests fail (expected)
- test_topic_separation.py: 15/15 tests fail (expected)

Next step: Implement features to make tests pass (TDD Green Phase)
EOF

git add IMPLEMENTATION_TRACKER.md test_baseline.log
git commit -m "Document TDD red phase baseline"
git push origin feature/testing-infrastructure
```

---

### **PHASE 2: Parallel Feature Implementation**

**Duration:** 2-3 weeks (wall-clock time)
**Agent Work:** 15-25 hours per agent (parallel)

#### **PHASE 2a: Agent Alpha - Bullet Quality** (15-20 hours)

**Branch:** `feature/bullet-quality-improvements`

**Tasks (in order):**

##### Task Alpha-1: Context-Aware Bullets (2-3 hours)

```bash
# Agent Alpha executes
git checkout feature/bullet-quality-improvements
git pull origin main  # Sync with latest

# Create implementation branch
git checkout -b alpha/context-aware-bullets
```

**Implementation Steps:**

1. **Modify `_content_to_slides` to track heading ancestry**

```python
# File: slide_generator_pkg/document_parser.py:1647

def _content_to_slides(self, content: str, fast_mode: bool = False) -> List[SlideContent]:
    """Convert script content to slides - each content block becomes one slide with bullet points"""
    # ... existing code ...

    # Track document hierarchy for smart subtitles
    current_h1 = None
    current_h2 = None
    current_h3 = None  # ADD THIS
    current_h4 = None  # ADD THIS

    # NEW: Build heading ancestry list
    def build_heading_ancestry():
        """Build list of current heading hierarchy"""
        ancestry = []
        if current_h1:
            ancestry.append(current_h1)
        if current_h2:
            ancestry.append(current_h2)
        if current_h3:
            ancestry.append(current_h3)
        if current_h4:
            ancestry.append(current_h4)
        return ancestry

    # ... in the loop where bullets are created ...

    # MODIFY EXISTING CALL (around line 1732):
    # OLD:
    # topic_sentence, bullet_points = self._create_bullet_points(
    #     combined_text, fast_mode, context_heading=temp_context
    # )

    # NEW:
    heading_ancestry = build_heading_ancestry()
    topic_sentence, bullet_points = self._create_bullet_points(
        combined_text,
        fast_mode,
        context_heading=temp_context,
        heading_ancestry=heading_ancestry  # NEW PARAMETER
    )
```

2. **Update `_create_bullet_points` signature**

```python
# File: slide_generator_pkg/document_parser.py:1951

def _create_bullet_points(
    self,
    text: str,
    fast_mode: bool = False,
    context_heading: str = None,
    heading_ancestry: List[str] = None  # NEW PARAMETER
) -> Tuple[Optional[str], List[str]]:
    """
    Create bullet points from text using the unified bullet strategy.

    Args:
        text: Source text
        fast_mode: Use faster basic extraction
        context_heading: Immediate heading context
        heading_ancestry: Full heading hierarchy (e.g., ["Intro", "Background", "Problem"])
    """
    # ... existing code ...

    # Pass ancestry to unified bullets
    bullets = self._create_unified_bullets(
        text,
        context_heading=context_heading,
        heading_ancestry=heading_ancestry  # NEW
    )
```

3. **Update `_create_unified_bullets` to pass ancestry to LLM**

```python
# File: slide_generator_pkg/document_parser.py:2009

def _create_unified_bullets(
    self,
    text: str,
    context_heading: str = None,
    heading_ancestry: List[str] = None  # NEW
) -> List[str]:
    # ... existing code ...

    # Step 4: Try LLM-based bullets
    if self.client or self.openai_client:
        try:
            llm_bullets = self._create_llm_only_bullets(
                text,
                context_heading=context_heading,
                heading_ancestry=heading_ancestry  # NEW
            )
```

4. **Update `_create_llm_only_bullets` to include ancestry in prompt**

```python
# File: slide_generator_pkg/document_parser.py:2945

def _create_llm_only_bullets(
    self,
    text: str,
    context_heading: str = None,
    heading_ancestry: List[str] = None,  # NEW
    enforce_diversity: bool = False
) -> List[str]:
    # ... existing code ...

    # Build context string
    context_parts = []

    if heading_ancestry and len(heading_ancestry) > 1:
        # Show hierarchy: "Introduction > Background > Problem Statement"
        hierarchy = " > ".join(heading_ancestry)
        context_parts.append(f"Document hierarchy: {hierarchy}")
        context_parts.append(f"Current section: {heading_ancestry[-1]}")
    elif context_heading:
        context_parts.append(f"Section: {context_heading}")

    context_note = "\n".join(context_parts) if context_parts else ""

    # Add to prompt (around line 2860)
    prompt = f"""Generate {bullet_count} concise bullet points for this slide.

{context_note}

CONTENT TYPE: {content_type}
STYLE: {style}

{structured_prompt}

SOURCE TEXT:
{text}

Generate exactly {bullet_count} bullets (8-15 words each):"""
```

5. **Run tests**

```bash
# Agent Alpha executes
python3 -m pytest tests/test_bullet_quality.py::TestContextAwareBullets -v

# Should now pass!
```

6. **Commit**

```bash
git add slide_generator_pkg/document_parser.py
git commit -m "Implement context-aware bullets (1.2)

- Add heading_ancestry parameter throughout bullet generation chain
- Pass full document hierarchy to LLM prompts
- Update _content_to_slides to track H1-H4 ancestry
- Tests: test_bullet_quality.py::TestContextAwareBullets now pass"

git push origin alpha/context-aware-bullets
```

##### Task Alpha-2: Bullet Validation (4-6 hours)

```bash
git checkout feature/bullet-quality-improvements
git checkout -b alpha/bullet-validation
```

**Implementation:**

1. **Add validation method**

```python
# File: slide_generator_pkg/document_parser.py (add after _create_llm_only_bullets)

def _validate_and_improve_bullets(
    self,
    bullets: List[str],
    source_text: str,
    heading: str,
    parent_headings: List[str] = None
) -> Tuple[List[str], dict]:
    """
    Validate bullet quality and improve if needed using LLM.

    Returns:
        (improved_bullets, metrics_dict)
    """
    if not self.client and not self.openai_client:
        # No LLM available, return as-is
        return bullets, {
            'relevance_score': 0.0,
            'completeness_score': 0.0,
            'missing_concepts': [],
            'improvements_made': 0
        }

    # Build context
    context = " > ".join(parent_headings) if parent_headings else ""

    # Create validation prompt
    bullets_text = "\n".join(f"• {b}" for b in bullets)

    prompt = f"""Review these slide bullets for quality and relevance.

SLIDE TITLE: {heading}
{f'CONTEXT: {context}' if context else ''}

SOURCE TEXT:
{source_text}

CURRENT BULLETS:
{bullets_text}

EVALUATE:
1. Relevance (0.0-1.0): Do bullets capture main points from source?
2. Completeness (0.0-1.0): Are key concepts missing?
3. Missing concepts: List important points not covered

If relevance or completeness < 0.8, provide improved bullets.

FORMAT:
Relevance: [0.0-1.0]
Completeness: [0.0-1.0]
Missing: [concept1, concept2, ...]
Improved Bullets:
• [bullet 1]
• [bullet 2]
..."""

    # Call LLM
    if self.client:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text
    else:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        response_text = response.choices[0].message.content

    # Parse response
    relevance = 0.5
    completeness = 0.5
    missing = []
    improved_bullets = bullets  # Default to original

    for line in response_text.split('\n'):
        line = line.strip()
        if line.lower().startswith('relevance:'):
            try:
                relevance = float(line.split(':')[1].strip())
            except:
                pass
        elif line.lower().startswith('completeness:'):
            try:
                completeness = float(line.split(':')[1].strip())
            except:
                pass
        elif line.lower().startswith('missing:'):
            missing_str = line.split(':', 1)[1].strip()
            missing = [m.strip() for m in missing_str.strip('[]').split(',')]

    # Extract improved bullets if provided
    if 'Improved Bullets:' in response_text:
        improved_section = response_text.split('Improved Bullets:')[1]
        improved_bullets = []
        for line in improved_section.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                bullet = line.lstrip('•-').strip()
                if bullet and len(bullet.split()) >= 4:
                    improved_bullets.append(bullet)

        # Only use improved bullets if we got valid ones
        if not improved_bullets:
            improved_bullets = bullets

    improvements_made = 1 if improved_bullets != bullets else 0

    metrics = {
        'relevance_score': relevance,
        'completeness_score': completeness,
        'missing_concepts': missing,
        'improvements_made': improvements_made
    }

    logger.info(f"Bullet validation: relevance={relevance:.2f}, completeness={completeness:.2f}, improved={improvements_made}")

    return improved_bullets, metrics
```

2. **Integrate validation into bullet generation**

```python
# File: slide_generator_pkg/document_parser.py:2945
# Modify _create_llm_only_bullets to call validation

def _create_llm_only_bullets(
    self,
    text: str,
    context_heading: str = None,
    heading_ancestry: List[str] = None,
    enforce_diversity: bool = False,
    validate_quality: bool = True  # NEW PARAMETER (default True)
) -> List[str]:
    # ... existing bullet generation code ...

    # Parse bullets
    bullets = self._parse_bullets(response_text)

    # NEW: Validate and improve if enabled
    if validate_quality and bullets:
        validated_bullets, metrics = self._validate_and_improve_bullets(
            bullets,
            text,
            context_heading or "Content",
            parent_headings=heading_ancestry
        )

        # Use validated bullets if quality improved
        if metrics['improvements_made'] > 0:
            logger.info(f"Bullets improved by validation (relevance: {metrics['relevance_score']:.2f})")
            bullets = validated_bullets

    return bullets
```

3. **Test**

```bash
python3 -m pytest tests/test_bullet_quality.py::TestBulletValidation -v
```

4. **Commit**

```bash
git add slide_generator_pkg/document_parser.py
git commit -m "Implement bullet validation and self-correction (1.1)

- Add _validate_and_improve_bullets method with LLM validation
- Integrate validation into _create_llm_only_bullets
- Track relevance, completeness, and missing concepts
- Auto-improve bullets scoring < 0.8
- Tests: test_bullet_quality.py::TestBulletValidation now pass"

git push origin alpha/bullet-validation
```

##### Task Alpha-3: Bullet Diversity (3-4 hours)

```bash
git checkout feature/bullet-quality-improvements
git checkout -b alpha/bullet-diversity
```

**Implementation:**

```python
# File: slide_generator_pkg/document_parser.py (add new method)

def _check_bullet_diversity(self, bullets: List[str]) -> float:
    """
    Score bullet diversity (0.0-1.0).
    Low score indicates repetitive structure.
    """
    if not bullets or len(bullets) < 2:
        return 1.0

    diversity_score = 0.0

    # 1. Starting word diversity (40% weight)
    start_words = []
    for bullet in bullets:
        words = bullet.split()
        if words:
            start_words.append(words[0].lower())

    if start_words:
        unique_starts = len(set(start_words))
        start_diversity = unique_starts / len(start_words)
        diversity_score += start_diversity * 0.4

    # 2. Length variance (30% weight)
    lengths = [len(b.split()) for b in bullets]
    if lengths and len(lengths) > 1:
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        # Normalize: variance of 10 = perfect (1.0), 0 = poor (0.0)
        length_diversity = min(variance / 10.0, 1.0)
        diversity_score += length_diversity * 0.3

    # 3. Structural diversity (30% weight)
    # Check if bullets follow different patterns
    patterns = []
    for bullet in bullets:
        # Identify pattern: starts with verb, noun, adjective, etc.
        words = bullet.split()
        if words:
            # Simple heuristic: first word POS
            first_word = words[0].lower()
            if first_word in ['use', 'create', 'build', 'implement', 'provide']:
                patterns.append('verb')
            elif first_word in ['the', 'a', 'an']:
                patterns.append('article')
            else:
                patterns.append('other')

    if patterns:
        unique_patterns = len(set(patterns))
        pattern_diversity = unique_patterns / len(patterns)
        diversity_score += pattern_diversity * 0.3

    return diversity_score

# Modify _create_llm_only_bullets to use diversity check
def _create_llm_only_bullets(
    self,
    text: str,
    context_heading: str = None,
    heading_ancestry: List[str] = None,
    enforce_diversity: bool = False,
    validate_quality: bool = True
) -> List[str]:
    # ... existing code to generate bullets ...

    bullets = self._parse_bullets(response_text)

    # NEW: Check diversity (before validation)
    if not enforce_diversity and bullets:
        diversity = self._check_bullet_diversity(bullets)

        if diversity < 0.5:
            logger.warning(f"Low bullet diversity ({diversity:.2f}), regenerating with diversity instruction")
            # Retry with diversity enforcement
            return self._create_llm_only_bullets(
                text,
                context_heading=context_heading,
                heading_ancestry=heading_ancestry,
                enforce_diversity=True,  # Flag to add diversity instruction
                validate_quality=validate_quality
            )

    # Add diversity instruction to prompt if enforced
    if enforce_diversity:
        # Modify prompt (around line 2860) to add:
        diversity_instruction = """
IMPORTANT: Make bullets DIVERSE:
- Vary starting words (don't repeat "The...", "This...", "It...")
- Use different sentence structures
- Vary bullet lengths (mix short and longer bullets)
"""
        # Insert this before "Generate exactly {bullet_count} bullets:"

    # ... rest of method (validation, return) ...
```

**Test & Commit:**

```bash
python3 -m pytest tests/test_bullet_quality.py::TestBulletDiversity -v

git add slide_generator_pkg/document_parser.py
git commit -m "Implement bullet diversity scoring (1.3)

- Add _check_bullet_diversity method
- Check starting words, length variance, structural patterns
- Auto-regenerate bullets with diversity < 0.5
- Add diversity instruction to LLM prompt
- Tests: test_bullet_quality.py::TestBulletDiversity now pass"

git push origin alpha/bullet-diversity
```

##### Task Alpha-4: Integration & Testing

```bash
# Merge all alpha branches into feature branch
git checkout feature/bullet-quality-improvements
git merge alpha/context-aware-bullets
git merge alpha/bullet-validation
git merge alpha/bullet-diversity

# Run full test suite
./scripts/quick_ci.sh

# Run specific tests
python3 -m pytest tests/test_bullet_quality.py -v

# Update tracker
cat >> IMPLEMENTATION_TRACKER.md << 'EOF'

## Agent Alpha Completion
**Date:** [timestamp]
**Status:** ✅ All bullet quality improvements complete

Completed:
- [x] 1.2 Context-Aware Bullets (3 hours)
- [x] 1.1 Bullet Validation (5 hours)
- [x] 1.3 Bullet Diversity (4 hours)

Test Results:
- test_bullet_quality.py: 12/12 tests PASS ✅
- Regression benchmark: +12% quality improvement

Ready for integration.
EOF

git add IMPLEMENTATION_TRACKER.md
git commit -m "Agent Alpha complete: All bullet quality improvements implemented"
git push origin feature/bullet-quality-improvements
```

---

#### **PHASE 2b: Agent Beta - Topic Separation** (18-25 hours)

**Branch:** `feature/topic-separation-improvements`

**Tasks (in order):**

##### Task Beta-1: Topic Boundary Detection (6-8 hours)

```bash
git checkout feature/topic-separation-improvements
git checkout -b beta/topic-boundaries
```

**Implementation:**

```python
# File: slide_generator_pkg/document_parser.py (add new methods)

def _detect_topic_boundaries(
    self,
    paragraphs: List[str],
    use_llm: bool = True,
    similarity_threshold: float = 0.75
) -> List[int]:
    """
    Detect where topics change in unstructured text.
    Returns indices where new topics begin.
    """
    if len(paragraphs) < 2:
        return [0]

    boundaries = [0]  # Always start at beginning

    # Method 1: Semantic embeddings (if OpenAI available)
    if self.openai_client:
        try:
            embeddings = []

            # Get embeddings for each paragraph
            for para in paragraphs:
                if len(para.strip()) < 10:
                    embeddings.append(None)
                    continue

                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=para[:8000]  # Truncate if needed
                )
                embeddings.append(response.data[0].embedding)

            # Calculate cosine similarity between consecutive paragraphs
            import numpy as np

            for i in range(1, len(paragraphs)):
                if embeddings[i-1] is None or embeddings[i] is None:
                    continue

                # Cosine similarity
                vec1 = np.array(embeddings[i-1])
                vec2 = np.array(embeddings[i])
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                # Low similarity = topic boundary
                if similarity < similarity_threshold:
                    boundaries.append(i)
                    logger.info(f"Topic boundary detected at paragraph {i} (similarity: {similarity:.2f})")

        except Exception as e:
            logger.warning(f"Embedding-based boundary detection failed: {e}")
            # Fall back to LLM validation

    # Method 2: LLM validation (refine boundaries or fallback)
    if use_llm and (self.client or self.openai_client) and len(boundaries) > 1:
        validated_boundaries = self._validate_topic_boundaries(paragraphs, boundaries)
        return validated_boundaries

    return boundaries if len(boundaries) > 1 else [0]

def _validate_topic_boundaries(
    self,
    paragraphs: List[str],
    proposed_boundaries: List[int]
) -> List[int]:
    """Use LLM to validate proposed topic boundaries"""
    validated = [0]

    for boundary_idx in proposed_boundaries[1:]:
        if boundary_idx >= len(paragraphs):
            continue

        # Get context around boundary
        before_idx = max(0, boundary_idx - 1)
        after_idx = min(len(paragraphs) - 1, boundary_idx)

        before_text = paragraphs[before_idx][:500]  # Truncate
        after_text = paragraphs[after_idx][:500]

        prompt = f"""Are these paragraphs about the SAME topic or DIFFERENT topics?

Paragraph A:
{before_text}

Paragraph B:
{after_text}

Answer ONLY: SAME or DIFFERENT"""

        try:
            if self.client:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text.strip().upper()
            else:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                answer = response.choices[0].message.content.strip().upper()

            if "DIFFERENT" in answer:
                validated.append(boundary_idx)
                logger.info(f"LLM confirmed topic boundary at paragraph {boundary_idx}")

        except Exception as e:
            logger.warning(f"LLM validation failed for boundary {boundary_idx}: {e}")
            # Keep boundary if embedding detected it
            validated.append(boundary_idx)

    return validated

# Integrate into _content_to_slides
def _content_to_slides(self, content: str, fast_mode: bool = False) -> List[SlideContent]:
    """Convert script content to slides - each content block becomes one slide with bullet points"""

    # ... existing code ...

    # NEW: Check if document has headings
    has_headings = any(line.strip().startswith('#') for line in lines)

    # If no headings, use topic boundary detection
    if not has_headings and len(lines) > 5:
        logger.info("No headings detected, using intelligent topic boundary detection")

        # Group lines into paragraphs
        paragraphs = []
        current_para = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append(' '.join(current_para))

        # Detect boundaries
        boundaries = self._detect_topic_boundaries(paragraphs)

        # Create slides from topic boundaries
        for i, boundary_idx in enumerate(boundaries):
            end_idx = boundaries[i+1] if i+1 < len(boundaries) else len(paragraphs)
            topic_text = "\n\n".join(paragraphs[boundary_idx:end_idx])

            # Generate bullets
            topic_sentence, bullet_points = self._create_bullet_points(topic_text, fast_mode)

            # Generate title from bullets
            slide_title = self._create_title_from_bullets(bullet_points, topic_text)

            slides.append(SlideContent(
                title=slide_title,
                content=bullet_points,
                slide_type='content'
            ))

        return slides

    # ... existing heading-based processing ...
```

**Test & Commit:**

```bash
python3 -m pytest tests/test_topic_separation.py::TestTopicBoundaryDetection -v

git add slide_generator_pkg/document_parser.py
git commit -m "Implement intelligent topic boundary detection (2.1)

- Add _detect_topic_boundaries using embeddings
- Add _validate_topic_boundaries using LLM
- Integrate into _content_to_slides for unstructured docs
- Use cosine similarity threshold of 0.75
- Tests: test_topic_separation.py::TestTopicBoundaryDetection pass"

git push origin beta/topic-boundaries
```

##### Task Beta-2: Semantic Clustering (5-7 hours)

```bash
git checkout feature/topic-separation-improvements
git checkout -b beta/semantic-clustering
```

**Implementation:**

```python
# File: slide_generator_pkg/document_parser.py (add methods)

def _create_semantic_topic_slides(self, text: str) -> List[SlideContent]:
    """
    Use semantic clustering to automatically group content into topic-based slides.
    """
    if not self.semantic_analyzer.initialized:
        logger.info("Semantic analyzer not available, skipping clustering")
        return None

    # Split into sentences
    sentences = self._split_into_sentences(text)

    if len(sentences) < 5:
        return None

    # Analyze chunks
    chunks = self.semantic_analyzer.analyze_chunks(sentences)

    # Cluster by topic
    topic_clusters = self._cluster_chunks_by_topic(chunks)

    if not topic_clusters:
        return None

    slides = []
    for cluster_id, cluster_chunks in topic_clusters.items():
        # Combine chunks
        cluster_text = " ".join([c.text for c in cluster_chunks])

        # Generate bullets
        _, bullet_points = self._create_bullet_points(cluster_text)

        if not bullet_points:
            continue

        # Generate topic title
        title = self._generate_topic_title(cluster_chunks, bullet_points)

        slides.append(SlideContent(
            title=title,
            content=bullet_points,
            slide_type='content'
        ))

    logger.info(f"Created {len(slides)} slides from semantic clustering")
    return slides

def _split_into_sentences(self, text: str) -> List[str]:
    """Split text into sentences"""
    # Use NLTK if available
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    except:
        # Fallback: split on periods
        sentences = text.split('.')
        return [s.strip() + '.' for s in sentences if len(s.strip()) > 10]

def _cluster_chunks_by_topic(
    self,
    chunks: List[SemanticChunk],
    min_cluster_size: int = 2,
    max_clusters: int = 10
) -> dict:
    """
    Cluster chunks by topic using embeddings.
    Returns: {cluster_id: [chunks]}
    """
    if not chunks or len(chunks) < min_cluster_size:
        return {}

    try:
        # Get embeddings from chunks or generate new ones
        if self.openai_client:
            embeddings = []
            for chunk in chunks:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk.text[:8000]
                )
                embeddings.append(response.data[0].embedding)

            # Cluster using K-means
            import numpy as np
            from sklearn.cluster import KMeans

            X = np.array(embeddings)

            # Determine optimal cluster count (2 to max_clusters)
            n_clusters = min(max_clusters, max(2, len(chunks) // 3))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Group chunks by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(chunks[i])

            # Filter out too-small clusters
            clusters = {
                k: v for k, v in clusters.items()
                if len(v) >= min_cluster_size
            }

            return clusters

        else:
            # No embeddings available, return single cluster
            return {0: chunks}

    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        return {0: chunks}

def _generate_topic_title(
    self,
    chunks: List[SemanticChunk],
    bullets: List[str]
) -> str:
    """Generate a concise topic title from cluster content"""

    combined_text = " ".join([c.text for c in chunks])[:1000]
    bullets_text = "\n".join(f"• {b}" for b in bullets[:3])

    if not self.client and not self.openai_client:
        # Fallback: extract first few words
        words = combined_text.split()
        return " ".join(words[:5]).title()

    prompt = f"""Generate a concise slide title (3-6 words max) for this content.

Bullets:
{bullets_text}

Context:
{combined_text[:300]}

Title:"""

    try:
        if self.client:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            title = response.content[0].text.strip()
        else:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            title = response.choices[0].message.content.strip()

        # Clean title
        title = title.strip('"\'').strip()

        # Limit length
        if len(title.split()) > 8:
            title = " ".join(title.split()[:6])

        return title

    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        # Fallback
        return "Topic " + str(len(bullets))

# Update _content_to_slides to try semantic clustering
def _content_to_slides(self, content: str, fast_mode: bool = False) -> List[SlideContent]:
    # ... existing code ...

    # NEW: Try semantic clustering before topic boundaries
    if not has_headings and len(lines) > 5:
        logger.info("Attempting semantic clustering for unstructured document")

        semantic_slides = self._create_semantic_topic_slides(content)

        if semantic_slides and len(semantic_slides) > 1:
            logger.info(f"Successfully created {len(semantic_slides)} slides via semantic clustering")
            return semantic_slides

        # Fall back to topic boundaries
        logger.info("Semantic clustering insufficient, using topic boundary detection")
        # ... existing topic boundary code ...
```

**File:** `slide_generator_pkg/semantic_analyzer.py` (add method)

```python
# Add to SemanticAnalyzer class

def cluster_by_topic(
    self,
    chunks: List[SemanticChunk],
    min_cluster_size: int = 3,
    max_clusters: int = 10
) -> dict:
    """
    Cluster chunks by semantic topic.
    Returns: {cluster_id: [chunks]}
    """
    if not self.initialized or not chunks:
        return {}

    try:
        if self.use_heavy_analysis and self.model:
            # Use sentence transformers
            texts = [c.text for c in chunks]
            embeddings = self.model.encode(texts)

            # K-means clustering
            from sklearn.cluster import KMeans
            import numpy as np

            n_clusters = min(max_clusters, max(2, len(chunks) // 3))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Group by label
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(chunks[i])

            # Filter small clusters
            clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

            return clusters

        else:
            # Lightweight fallback: single cluster
            return {0: chunks} if len(chunks) >= min_cluster_size else {}

    except Exception as e:
        logging.warning(f"Clustering failed: {e}")
        return {}
```

**Test & Commit:**

```bash
python3 -m pytest tests/test_topic_separation.py::TestSemanticClustering -v

git add slide_generator_pkg/document_parser.py slide_generator_pkg/semantic_analyzer.py
git commit -m "Implement semantic clustering for topic grouping (2.2)

- Add _create_semantic_topic_slides method
- Add _cluster_chunks_by_topic using K-means
- Add _generate_topic_title with LLM
- Add cluster_by_topic to SemanticAnalyzer
- Integrate into _content_to_slides
- Tests: test_topic_separation.py::TestSemanticClustering pass"

git push origin beta/semantic-clustering
```

##### Task Beta-3: Smart Content Splitting (4-5 hours)

```bash
git checkout feature/topic-separation-improvements
git checkout -b beta/smart-splitting
```

**Implementation:**

```python
# File: slide_generator_pkg/document_parser.py (add methods)

def _split_large_content_block(
    self,
    text: str,
    heading: str,
    max_bullets_per_slide: int = 5
) -> List[SlideContent]:
    """
    Split large content blocks into multiple slides if needed.
    """
    # Generate all bullets first
    _, all_bullets = self._create_bullet_points(text, context_heading=heading)

    if not all_bullets:
        return []

    # If bullets fit on one slide, done
    if len(all_bullets) <= max_bullets_per_slide:
        return [SlideContent(
            title=heading,
            content=all_bullets,
            slide_type='content'
        )]

    # Need to split - group bullets by sub-topic
    logger.info(f"Splitting {len(all_bullets)} bullets into multiple slides (max {max_bullets_per_slide} per slide)")

    sub_topics = self._group_bullets_by_subtopic(all_bullets, heading)

    slides = []
    for i, (subtopic_name, subtopic_bullets) in enumerate(sub_topics.items()):
        # First slide uses main heading, subsequent use sub-topic names
        if i == 0:
            slide_title = heading
        else:
            slide_title = f"{heading}: {subtopic_name}"

        slides.append(SlideContent(
            title=slide_title,
            content=subtopic_bullets,
            slide_type='content'
        ))

    logger.info(f"Split into {len(slides)} slides")
    return slides

def _group_bullets_by_subtopic(
    self,
    bullets: List[str],
    main_heading: str
) -> dict:
    """
    Group bullets into logical sub-topics using LLM.
    Returns: {subtopic_name: [bullets]}
    """
    if not bullets:
        return {}

    if not self.client and not self.openai_client:
        # Fallback: simple chunking
        chunk_size = 5
        chunks = {}
        for i in range(0, len(bullets), chunk_size):
            chunk_bullets = bullets[i:i+chunk_size]
            chunks[f"Part {i//chunk_size + 1}"] = chunk_bullets
        return chunks

    # Use LLM to intelligently group
    bullets_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))

    prompt = f"""Group these bullets into 2-3 logical sub-topics.

Main Topic: {main_heading}

Bullets:
{bullets_text}

Provide sub-topic groups. Format:

Sub-topic: [Name]
Bullets: [1, 3, 5]

Sub-topic: [Name]
Bullets: [2, 4, 6]
"""

    try:
        if self.client:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
        else:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            response_text = response.choices[0].message.content

        # Parse response
        grouped = {}
        current_topic = None

        for line in response_text.split('\n'):
            line = line.strip()

            if line.lower().startswith('sub-topic:'):
                current_topic = line.split(':', 1)[1].strip()
                grouped[current_topic] = []

            elif line.lower().startswith('bullets:') and current_topic:
                # Extract bullet numbers
                bullet_nums_str = line.split(':', 1)[1].strip()
                # Parse: "1, 3, 5" or "[1, 3, 5]"
                bullet_nums_str = bullet_nums_str.strip('[]')
                bullet_indices = []

                for num_str in bullet_nums_str.split(','):
                    try:
                        idx = int(num_str.strip()) - 1  # Convert to 0-indexed
                        if 0 <= idx < len(bullets):
                            bullet_indices.append(idx)
                    except:
                        pass

                # Add bullets to this sub-topic
                for idx in bullet_indices:
                    grouped[current_topic].append(bullets[idx])

        # Validate grouping
        if grouped and sum(len(v) for v in grouped.values()) >= len(bullets) * 0.8:
            # Good grouping (covers 80%+ of bullets)
            return grouped
        else:
            # Poor grouping, fall back to chunking
            raise ValueError("LLM grouping covered < 80% of bullets")

    except Exception as e:
        logger.warning(f"LLM grouping failed: {e}, using simple chunking")
        # Fallback: chunk by 5
        chunks = {}
        chunk_size = 5
        for i in range(0, len(bullets), chunk_size):
            chunk_bullets = bullets[i:i+chunk_size]
            chunks[f"Part {i//chunk_size + 1}"] = chunk_bullets
        return chunks

# Integrate into _content_to_slides
def _content_to_slides(self, content: str, fast_mode: bool = False) -> List[SlideContent]:
    # ... existing code ...

    # When creating slide from content block:
    # OLD:
    # topic_sentence, bullet_points = self._create_bullet_points(combined_text, ...)
    # slides.append(SlideContent(title=..., content=bullet_points, ...))

    # NEW:
    topic_sentence, bullet_points = self._create_bullet_points(combined_text, ...)

    # Check if we need to split
    if len(bullet_points) > 6:  # Threshold for splitting
        split_slides = self._split_large_content_block(
            combined_text,
            slide_title,
            max_bullets_per_slide=5
        )
        slides.extend(split_slides)
    else:
        slides.append(SlideContent(
            title=slide_title,
            content=bullet_points,
            slide_type='content'
        ))
```

**Test & Commit:**

```bash
python3 -m pytest tests/test_topic_separation.py::TestSmartContentSplitting -v

git add slide_generator_pkg/document_parser.py
git commit -m "Implement smart content splitting (2.3)

- Add _split_large_content_block method
- Add _group_bullets_by_subtopic using LLM
- Auto-split slides with > 6 bullets
- Create sub-topic slides with descriptive titles
- Tests: test_topic_separation.py::TestSmartContentSplitting pass"

git push origin beta/smart-splitting
```

##### Task Beta-4: Integration & Testing

```bash
# Merge beta branches
git checkout feature/topic-separation-improvements
git merge beta/topic-boundaries
git merge beta/semantic-clustering
git merge beta/smart-splitting

# Run tests
./scripts/quick_ci.sh
python3 -m pytest tests/test_topic_separation.py -v

# Update tracker
cat >> IMPLEMENTATION_TRACKER.md << 'EOF'

## Agent Beta Completion
**Date:** [timestamp]
**Status:** ✅ All topic separation improvements complete

Completed:
- [x] 2.1 Topic Boundaries (7 hours)
- [x] 2.2 Semantic Clustering (6 hours)
- [x] 2.3 Smart Splitting (5 hours)

Test Results:
- test_topic_separation.py: 15/15 tests PASS ✅
- Integration tests: PASS ✅

Ready for integration.
EOF

git add IMPLEMENTATION_TRACKER.md
git commit -m "Agent Beta complete: All topic separation improvements implemented"
git push origin feature/topic-separation-improvements
```

---

### **PHASE 3: Integration & Merge** (Agent Delta - 4-6 hours)

**Objective:** Merge feature branches, resolve conflicts, validate integration

#### Step 3.1: Pull Latest Changes

```bash
# Agent Delta executes
git checkout main
git pull origin main

# Pull feature branches
git fetch origin feature/bullet-quality-improvements
git fetch origin feature/topic-separation-improvements
git fetch origin feature/testing-infrastructure
```

#### Step 3.2: Merge Testing Infrastructure First

```bash
git checkout main
git merge origin/feature/testing-infrastructure

# Run tests (should fail on new tests, pass on existing)
./scripts/quick_ci.sh

# Commit
git push origin main
```

#### Step 3.3: Create Integration Branch

```bash
git checkout -b integration/phase-2-features
```

#### Step 3.4: Merge Bullet Quality

```bash
git merge origin/feature/bullet-quality-improvements

# Resolve conflicts if any
# ... manual conflict resolution ...

# Test
./scripts/quick_ci.sh
python3 -m pytest tests/test_bullet_quality.py -v

# If pass, continue
```

#### Step 3.5: Merge Topic Separation

```bash
git merge origin/feature/topic-separation-improvements

# Resolve conflicts (likely in document_parser.py)
# Common conflicts:
# - _content_to_slides method (both branches modify it)
# - Method signatures

# Resolution strategy:
# 1. Keep both feature sets
# 2. Ensure heading_ancestry flows through bullet generation
# 3. Ensure topic boundaries + semantic clustering work together

# Test
./scripts/quick_ci.sh
python3 -m pytest tests/ -v

# Run integration tests
python3 -m pytest tests/test_integration.py -v
```

#### Step 3.6: Full Regression Testing

```bash
# Run regression benchmark
python3 tests/regression_benchmark.py --version phase2_integrated

# Compare to baseline
python3 tests/regression_benchmark.py --compare baseline phase2_integrated

# Expected improvements:
# - Overall quality: +15-20%
# - Topic separation: +25-30%
# - Bullet relevance: +18-22%
```

#### Step 3.7: Merge to Main

```bash
# If all tests pass
git checkout main
git merge integration/phase-2-features

# Final CI check
./scripts/quick_ci.sh

# If pass, push
git push origin main

# Update tracker
cat >> IMPLEMENTATION_TRACKER.md << 'EOF'

## Phase 2 Integration Complete
**Date:** [timestamp]
**Status:** ✅ All features merged to main

Merged Features:
- Bullet Quality (1.1, 1.2, 1.3)
- Topic Separation (2.1, 2.2, 2.3)

Test Results:
- All tests: 27/27 PASS ✅
- CI: PASS ✅
- Regression: +18% quality improvement ✅

Phase 2 complete. Ready for Phase 4 (UX enhancements).
EOF

git add IMPLEMENTATION_TRACKER.md
git commit -m "Phase 2 integration complete: All core improvements merged"
git push origin main
```

---

### **PHASE 4: UX Enhancements** (Agents Alpha + Beta - Parallel, 8-12 hours)

**Duration:** 1 week (wall-clock)

#### **Task 4a: Agent Alpha - Quality Metrics UI** (3-4 hours)

```bash
git checkout main
git pull
git checkout -b feature/quality-metrics-ui
```

**Implementation:**

1. **Add quality metrics to slide generation**

```python
# File: slide_generator_pkg/document_parser.py

# Modify SlideContent to include metrics
@dataclass
class SlideContent:
    title: str
    content: List[str]
    slide_type: str = 'content'
    heading_level: Optional[int] = None
    subheader: Optional[str] = None
    visual_cues: Optional[List[str]] = None
    quality_metrics: Optional[dict] = None  # NEW

# Update _create_bullet_points to return metrics
def _create_bullet_points(
    self,
    text: str,
    fast_mode: bool = False,
    context_heading: str = None,
    heading_ancestry: List[str] = None
) -> Tuple[Optional[str], List[str], dict]:  # NEW: return metrics
    # ... existing code ...

    # Capture metrics from validation
    metrics = {
        'bullet_count': len(bullets),
        'avg_bullet_length': sum(len(b.split()) for b in bullets) / len(bullets) if bullets else 0,
        'relevance_score': 0.0,
        'completeness_score': 0.0
    }

    # If validation ran, include those metrics
    if hasattr(self, '_last_validation_metrics'):
        metrics.update(self._last_validation_metrics)

    return topic_sentence, bullets, metrics

# Store metrics in _validate_and_improve_bullets
def _validate_and_improve_bullets(...):
    # ... existing code ...

    # Store for later retrieval
    self._last_validation_metrics = metrics

    return improved_bullets, metrics
```

2. **Update file_to_slides.py to include metrics**

```python
# File: file_to_slides.py

# In the /convert endpoint, capture metrics
@app.route('/convert', methods=['POST'])
def convert_document():
    # ... existing code ...

    # After doc_structure = parser.parse_file(...)

    # Extract quality metrics from slides
    slide_metrics = []
    for slide in doc_structure.slides:
        if hasattr(slide, 'quality_metrics') and slide.quality_metrics:
            slide_metrics.append({
                'title': slide.title,
                'metrics': slide.quality_metrics
            })

    # Return metrics in response
    return jsonify({
        'status': 'success',
        'pptx_file': output_filename,
        'slide_count': len(doc_structure.slides),
        'quality_metrics': slide_metrics  # NEW
    })
```

3. **Update HTML template to show metrics**

```html
<!-- File: templates/file_to_slides.html -->

<!-- Add metrics display section -->
<div id="quality-metrics-section" style="display:none; margin-top: 20px;">
    <h3>📊 Quality Metrics</h3>
    <div id="metrics-summary">
        <p><strong>Total Slides:</strong> <span id="total-slides">0</span></p>
        <p><strong>Average Bullet Quality:</strong> <span id="avg-quality">0%</span></p>
        <p><strong>Average Bullets per Slide:</strong> <span id="avg-bullets">0</span></p>
    </div>

    <h4>Per-Slide Details</h4>
    <div id="slide-metrics-list"></div>
</div>

<script>
// Update processDocument to show metrics
function processDocument() {
    // ... existing code ...

    fetch('/convert', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // ... existing success handling ...

            // NEW: Display quality metrics
            if (data.quality_metrics && data.quality_metrics.length > 0) {
                displayQualityMetrics(data.quality_metrics, data.slide_count);
            }
        }
    });
}

function displayQualityMetrics(metrics, slideCount) {
    const metricsSection = document.getElementById('quality-metrics-section');
    const metricsList = document.getElementById('slide-metrics-list');

    // Show section
    metricsSection.style.display = 'block';

    // Summary
    document.getElementById('total-slides').textContent = slideCount;

    const avgRelevance = metrics.reduce((sum, m) => sum + (m.metrics.relevance_score || 0), 0) / metrics.length;
    document.getElementById('avg-quality').textContent = (avgRelevance * 100).toFixed(1) + '%';

    const avgBullets = metrics.reduce((sum, m) => sum + (m.metrics.bullet_count || 0), 0) / metrics.length;
    document.getElementById('avg-bullets').textContent = avgBullets.toFixed(1);

    // Per-slide details
    metricsList.innerHTML = '';
    metrics.forEach((slideMetric, i) => {
        const slideDiv = document.createElement('div');
        slideDiv.className = 'slide-metric-card';
        slideDiv.style.cssText = 'border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;';

        const relevance = slideMetric.metrics.relevance_score || 0;
        const completeness = slideMetric.metrics.completeness_score || 0;
        const qualityColor = relevance >= 0.8 ? '#28a745' : (relevance >= 0.6 ? '#ffc107' : '#dc3545');

        slideDiv.innerHTML = `
            <h5>${i + 1}. ${slideMetric.title}</h5>
            <p>
                <span style="color: ${qualityColor}; font-weight: bold;">
                    Quality: ${(relevance * 100).toFixed(0)}%
                </span> |
                Bullets: ${slideMetric.metrics.bullet_count || 0} |
                Avg Length: ${(slideMetric.metrics.avg_bullet_length || 0).toFixed(1)} words
            </p>
            ${completeness < 0.8 && slideMetric.metrics.missing_concepts ?
                `<p style="color: #856404; background: #fff3cd; padding: 5px; border-radius: 3px;">
                    ⚠️ May be missing: ${slideMetric.metrics.missing_concepts.join(', ')}
                </p>` : ''}
        `;

        metricsList.appendChild(slideDiv);
    });
}
</script>

<style>
.slide-metric-card:hover {
    background-color: #f8f9fa;
}
</style>
```

**Test & Commit:**

```bash
# Test locally
python3 wsgi.py
# Upload a test document, verify metrics display

git add file_to_slides.py templates/file_to_slides.html slide_generator_pkg/document_parser.py
git commit -m "Add quality metrics to UI (3.1)

- Include quality scores in slide generation
- Display metrics in web interface
- Show per-slide quality breakdown
- Highlight low-quality slides for review"

git push origin feature/quality-metrics-ui
```

---

#### **Task 4b: Agent Beta - Bullet Regeneration** (2-3 hours)

```bash
git checkout main
git pull
git checkout -b feature/bullet-regeneration
```

**Implementation:**

1. **Add regeneration endpoint**

```python
# File: file_to_slides.py

@app.route('/regenerate-bullets', methods=['POST'])
def regenerate_bullets():
    """Regenerate bullets for a specific slide"""
    try:
        data = request.json
        slide_index = data.get('slide_index')
        source_text = data.get('source_text')
        heading = data.get('heading')

        # Initialize parser
        claude_key = session.get('claude_api_key')
        openai_key = session.get('openai_api_key')
        parser = DocumentParser(claude_api_key=claude_key, openai_api_key=openai_key)

        # Regenerate bullets
        _, new_bullets, metrics = parser._create_bullet_points(
            source_text,
            context_heading=heading
        )

        return jsonify({
            'status': 'success',
            'bullets': new_bullets,
            'metrics': metrics
        })

    except Exception as e:
        logger.error(f"Bullet regeneration failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
```

2. **Update UI with regeneration button**

```html
<!-- File: templates/file_to_slides.html -->

<!-- Modify slide metric card to include regenerate button -->
<script>
function displayQualityMetrics(metrics, slideCount) {
    // ... existing code ...

    slideDiv.innerHTML = `
        <h5>${i + 1}. ${slideMetric.title}</h5>
        <p>
            <span style="color: ${qualityColor}; font-weight: bold;">
                Quality: ${(relevance * 100).toFixed(0)}%
            </span> |
            Bullets: ${slideMetric.metrics.bullet_count || 0}
        </p>
        ${relevance < 0.8 ? `
            <button onclick="regenerateBullets(${i}, '${slideMetric.title}')"
                    class="btn btn-sm btn-warning">
                🔄 Regenerate Bullets
            </button>
        ` : ''}
    `;
}

async function regenerateBullets(slideIndex, heading) {
    const button = event.target;
    button.disabled = true;
    button.textContent = '⏳ Regenerating...';

    try {
        // Note: We'd need to store source text per slide
        // For now, prompt user to re-upload or show limitation

        const response = await fetch('/regenerate-bullets', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                slide_index: slideIndex,
                heading: heading,
                source_text: ''  // Would need to store this
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`Regenerated bullets:\n\n${data.bullets.join('\n• ')}`);
            // Refresh metrics
            location.reload();
        }
    } catch (error) {
        alert('Regeneration failed: ' + error.message);
    } finally {
        button.disabled = false;
        button.textContent = '🔄 Regenerate Bullets';
    }
}
</script>
```

**Test & Commit:**

```bash
git add file_to_slides.py templates/file_to_slides.html
git commit -m "Add bullet regeneration feature (3.3)

- Add /regenerate-bullets endpoint
- Add regenerate button for low-quality slides
- Allow users to retry bullet generation
- Show updated bullets and metrics"

git push origin feature/bullet-regeneration
```

---

#### **Integration & Merge**

```bash
# Agent Delta merges UX features
git checkout main
git merge feature/quality-metrics-ui
git merge feature/bullet-regeneration

# Test
./scripts/quick_ci.sh

# Push
git push origin main
```

---

### **PHASE 5: Final Validation & Deploy** (Agent Gamma + Delta - 2-4 hours)

#### Step 5.1: Full Regression Testing

```bash
# Agent Gamma executes comprehensive tests
python3 tests/regression_benchmark.py --version final_release

# Compare to baseline
python3 tests/regression_benchmark.py --compare baseline final_release

# Expected results:
# - Overall quality: +18-22%
# - Bullet quality: +15-20%
# - Topic separation: +25-30%
# - User experience: Significantly improved
```

#### Step 5.2: Update Documentation

```bash
# Agent Delta updates docs
cat >> README.md << 'EOF'

## Recent Improvements (Version 2.0)

### Bullet Quality Enhancements
- ✅ Context-aware bullets reference document hierarchy
- ✅ LLM validation catches missing concepts
- ✅ Diversity scoring prevents repetitive structure

### Topic Separation Improvements
- ✅ Intelligent boundary detection for unstructured docs
- ✅ Semantic clustering groups related content
- ✅ Smart splitting prevents overwhelming slides

### UX Features
- ✅ Quality metrics displayed per slide
- ✅ Bullet regeneration for low-quality slides

EOF
```

#### Step 5.3: Final CI Check

```bash
# Before merging PR, run CI
./scripts/quick_ci.sh

# Must pass all tests
```

#### Step 5.4: Create Release PR

```bash
git checkout -b release/v2.0-improvements

# Update version
echo "2.0.0" > VERSION

git add .
git commit -m "Release v2.0: Bullet quality and topic separation improvements

Summary of changes:
- 6 core feature improvements (1.1, 1.2, 1.3, 2.1, 2.2, 2.3)
- 2 UX enhancements (3.1, 3.3)
- Comprehensive test suite (42 new tests)
- Updated CI pipeline
- Quality improvement: +18-22%

See IMPROVEMENT_RECOMMENDATIONS.md for details."

git push origin release/v2.0-improvements

# Create PR via gh CLI
gh pr create --title "Release v2.0: Major Quality Improvements" \
             --body "$(cat IMPLEMENTATION_TRACKER.md)" \
             --base main \
             --head release/v2.0-improvements
```

#### Step 5.5: Merge & Deploy

```bash
# After PR review and approval
gh pr merge --squash

# Pull merged changes
git checkout main
git pull origin main

# Deploy to production (Heroku)
git push heroku main

# Verify deployment
heroku logs --tail
```

---

## 📊 Execution Timeline

### Wall-Clock Timeline (Parallel Execution)

| Phase | Duration | Agents | Work Hours |
|-------|----------|--------|------------|
| Phase 0: Setup | 1 hour | Delta | 1 hour |
| Phase 1: Tests | 6-8 hours | Gamma | 6-8 hours |
| Phase 2a: Alpha | 15-20 hours | Alpha | 15-20 hours |
| Phase 2b: Beta | 18-25 hours | Beta | 18-25 hours |
| Phase 3: Integration | 4-6 hours | Delta | 4-6 hours |
| Phase 4: UX | 8-12 hours | Alpha + Beta | 5-7 hours each |
| Phase 5: Deploy | 2-4 hours | Gamma + Delta | 2-4 hours |

**Total Wall-Clock Time:** ~2-3 weeks
**Total Agent Work Hours:** 65-90 hours (across 4 agents)
**Effective Speedup:** ~3-4x vs sequential implementation

---

## 🔄 Coordination Protocol

### Sync Points (Every 4-6 hours)

**Agents update `IMPLEMENTATION_TRACKER.md` with:**
- Current task status
- Completed features
- Test results
- Blockers
- Next steps

### Communication Channels

1. **IMPLEMENTATION_TRACKER.md** - Central progress tracking
2. **Git branches** - Feature isolation and parallel work
3. **CI results** - Automated quality gates

### Conflict Resolution

**When agents modify the same files:**

1. **Agent Delta** coordinates merge
2. **Strategy:**
   - Feature branches isolated until Phase 3
   - Integration branch for conflict resolution
   - Both feature sets preserved
   - Tests validate integration

### Testing Coordination

**Agent Gamma monitors:**
- All test runs across agents
- CI pipeline status
- Regression benchmarks
- Quality gates

---

## ✅ Success Criteria

### Code Quality
- [ ] All 42 new tests pass
- [ ] CI script passes
- [ ] No regressions in existing functionality
- [ ] Code coverage ≥ baseline

### Performance
- [ ] Overall quality improvement: +15% minimum
- [ ] Bullet relevance: +10% minimum
- [ ] Topic separation accuracy: +20% minimum

### Integration
- [ ] All feature branches merged without conflicts
- [ ] No duplicate code
- [ ] Consistent API signatures
- [ ] Documentation updated

### Deployment
- [ ] Production deployment successful
- [ ] No errors in production logs
- [ ] User-facing features functional

---

## 🚨 Risk Mitigation

### Risk: Merge Conflicts

**Likelihood:** High (both agents modify `document_parser.py`)

**Mitigation:**
- Feature branches isolated
- Agent Delta dedicated to integration
- Test-driven development catches integration issues early

### Risk: Test Failures

**Likelihood:** Medium

**Mitigation:**
- Write tests first (TDD red phase)
- Continuous testing during development
- Agent Gamma monitors all test runs

### Risk: Performance Degradation

**Likelihood:** Low

**Mitigation:**
- Regression benchmarks track performance
- Caching strategies maintained
- LLM calls optimized (validation optional)

### Risk: API Rate Limits

**Likelihood:** Medium (increased LLM calls)

**Mitigation:**
- Validation is optional (can disable)
- Caching reduces duplicate calls
- Retry logic with exponential backoff

---

## 📚 Resources

### Documentation
- `IMPROVEMENT_RECOMMENDATIONS.md` - Feature details
- `IMPLEMENTATION_TRACKER.md` - Live progress tracking
- `BULLET_GENERATION_ANALYSIS.md` - System architecture
- `OPENAI_INTEGRATION.md` - Dual-LLM setup

### Testing
- `tests/test_bullet_quality.py` - Bullet quality tests
- `tests/test_topic_separation.py` - Topic separation tests
- `tests/regression_benchmark.py` - Quality benchmarks
- `scripts/quick_ci.sh` - CI pipeline

### Git Branches
- `main` - Production code
- `feature/bullet-quality-improvements` - Alpha work
- `feature/topic-separation-improvements` - Beta work
- `feature/testing-infrastructure` - Gamma work
- `integration/phase-2-features` - Delta coordination

---

## 🎯 Next Steps to Execute This Plan

To start implementation with Claude Code Web:

1. **Open Claude Code Web** in your browser
2. **Navigate to project:** `/Users/summerrae/claude_code/slidegenerator`
3. **Run:** Start with Phase 0 (Agent Delta - Setup)
4. **Spawn agents in parallel:**
   - Agent Gamma: Phase 1 (Test Creation)
   - Agent Alpha: Phase 2a (Bullet Quality) - after Phase 1
   - Agent Beta: Phase 2b (Topic Separation) - after Phase 1

5. **Monitor progress** via `IMPLEMENTATION_TRACKER.md`

---

**Last Updated:** 2025-11-18
**Status:** Ready for execution
**Estimated Completion:** 2-3 weeks with parallel agents
