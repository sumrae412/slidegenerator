"""
Bullet Quality Metrics - Objective evaluation system

Provides quantitative scoring for generated bullets across multiple dimensions:
- Structure (count, length, parallel formatting)
- Relevance (keyword coverage, semantic similarity)
- Style (tone, voice, terminology)
- Readability (complexity, clarity)
"""

import re
from typing import List, Dict
import textstat


class BulletQualityMetrics:
    """
    Comprehensive quality scoring for bullet points
    """

    def evaluate(self, generated_bullets: List[str], test_case: dict) -> dict:
        """
        Evaluate bullets across multiple quality dimensions

        Args:
            generated_bullets: List of generated bullet points
            test_case: Test case dict with input_text, expected_bullets, quality_criteria

        Returns:
            dict with scores for each dimension and overall quality
        """
        if not generated_bullets:
            return self._empty_result()

        metrics = {}

        # METRIC 1: Structural Quality (0-100)
        metrics['structure_score'] = self._score_structure(
            generated_bullets,
            test_case.get('quality_criteria', {})
        )

        # METRIC 2: Content Relevance (0-100)
        metrics['relevance_score'] = self._score_relevance(
            generated_bullets,
            test_case['input_text'],
            test_case.get('context_heading')
        )

        # METRIC 3: Stylistic Consistency (0-100)
        metrics['style_score'] = self._score_style(
            generated_bullets,
            test_case.get('expected_style', 'professional')
        )

        # METRIC 4: Readability (0-100)
        metrics['readability_score'] = self._score_readability(generated_bullets)

        # METRIC 5: Golden Reference Similarity (if available)
        if 'expected_bullets' in test_case:
            metrics['golden_similarity'] = self._compare_to_golden(
                generated_bullets,
                test_case['expected_bullets']
            )

        # COMPOSITE SCORE (weighted average)
        metrics['overall_quality'] = (
            metrics['structure_score'] * 0.25 +
            metrics['relevance_score'] * 0.35 +
            metrics['style_score'] * 0.20 +
            metrics['readability_score'] * 0.20
        )

        # Add detailed breakdown
        metrics['bullet_count'] = len(generated_bullets)
        metrics['avg_word_count'] = sum(len(b.split()) for b in generated_bullets) / len(generated_bullets)
        metrics['avg_char_length'] = sum(len(b) for b in generated_bullets) / len(generated_bullets)

        return metrics

    def _score_structure(self, bullets: List[str], criteria: dict) -> float:
        """
        Score structural quality (count, length, parallel structure)
        """
        score = 100.0

        # Bullet count penalty
        min_bullets = criteria.get('min_bullets', 3)
        max_bullets = criteria.get('max_bullets', 5)

        if len(bullets) < min_bullets:
            score -= 20 * (min_bullets - len(bullets))
        elif len(bullets) > max_bullets:
            score -= 10 * (len(bullets) - max_bullets)

        # Length distribution
        word_counts = [len(b.split()) for b in bullets]
        avg_words = sum(word_counts) / len(word_counts)

        target_range = criteria.get('avg_word_length', (8, 15))
        if avg_words < target_range[0]:
            score -= 15  # Too short
        elif avg_words > target_range[1]:
            score -= 10  # Too long

        # Word count variance (should be consistent)
        if len(word_counts) > 1:
            variance = sum((w - avg_words) ** 2 for w in word_counts) / len(word_counts)
            if variance > 25:  # High variance = inconsistent
                score -= 10

        # Parallel structure check (first word POS tags should be similar)
        try:
            first_words = [b.split()[0].lower() for b in bullets if b]

            # Check if most bullets start with similar patterns
            # Simple heuristic: common prefixes or word types
            if len(first_words) >= 3:
                # Count unique starting patterns
                unique_starts = len(set(first_words))
                if unique_starts == len(first_words):  # All different
                    score -= 10  # Likely not parallel
        except:
            pass

        # Check for proper punctuation
        for bullet in bullets:
            if not bullet.strip():
                score -= 10
            if bullet.endswith(('...', '..', 'such as', 'including')):
                score -= 5  # Incomplete

        # TRANSCRIPT-SPECIFIC QUALITY CHECKS
        # Check for conversational filler (transcript extraction quality)
        if 'must_not_contain_filler' in criteria:
            filler_penalty = self._check_conversational_filler(
                bullets,
                criteria['must_not_contain_filler']
            )
            score -= filler_penalty

        # Check for sentence truncation
        if criteria.get('no_truncation', False):
            truncation_penalty = self._check_sentence_truncation(bullets)
            score -= truncation_penalty

        # Check for complete sentences (must start and end properly)
        if criteria.get('must_be_complete_sentences', False):
            completeness_penalty = self._check_sentence_completeness(bullets)
            score -= completeness_penalty

        return max(0, min(100, score))

    def _score_relevance(self, bullets: List[str], input_text: str, context_heading: str = None) -> float:
        """
        Score content relevance (keyword coverage, semantic similarity)
        """
        score = 100.0

        bullets_text = ' '.join(bullets).lower()
        input_lower = input_text.lower()

        # Extract important keywords from input (simple TF-IDF-like approach)
        input_words = set(re.findall(r'\b[a-z]{4,}\b', input_lower))

        # Remove common stop words
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'will', 'would',
                      'could', 'should', 'their', 'there', 'these', 'those', 'what',
                      'which', 'when', 'where', 'about', 'more', 'some', 'such'}
        input_words = input_words - stop_words

        if not input_words:
            return 50.0  # Can't evaluate

        # Check keyword coverage
        bullets_words = set(re.findall(r'\b[a-z]{4,}\b', bullets_text))
        coverage = len(input_words & bullets_words) / len(input_words)

        if coverage < 0.2:
            score -= 30  # Very poor coverage
        elif coverage < 0.4:
            score -= 15  # Low coverage

        # Check for context heading keywords if provided
        if context_heading:
            heading_words = set(re.findall(r'\b[a-z]{3,}\b', context_heading.lower())) - stop_words
            if heading_words:
                heading_coverage = len(heading_words & bullets_words) / len(heading_words)
                if heading_coverage < 0.3:
                    score -= 10  # Not aligned with heading

        # Penalize if bullets are too generic (lots of filler words)
        filler_count = sum(1 for word in ['various', 'multiple', 'several', 'many', 'numerous']
                          if word in bullets_text)
        if filler_count > 2:
            score -= 5 * filler_count

        return max(0, min(100, score))

    def _score_style(self, bullets: List[str], expected_style: str) -> float:
        """
        Score stylistic consistency (tone, voice, terminology)
        """
        score = 100.0

        bullets_text = ' '.join(bullets).lower()

        # Style indicators
        style_patterns = {
            'educational': {
                'keywords': ['learn', 'students', 'course', 'understand', 'apply', 'concepts'],
                'verbs': ['learn', 'study', 'practice', 'master', 'explore']
            },
            'technical': {
                'keywords': ['system', 'implementation', 'architecture', 'processes', 'algorithm'],
                'verbs': ['executes', 'processes', 'manages', 'handles', 'implements']
            },
            'executive': {
                'keywords': ['costs', 'revenue', 'growth', 'strategy', 'results', 'performance'],
                'metrics': r'\d+%|\$[\d,]+|Q[1-4]'
            },
            'professional': {
                'keywords': ['provides', 'enables', 'organizations', 'companies', 'benefits'],
                'verbs': ['provides', 'enables', 'improves', 'reduces', 'enhances']
            }
        }

        patterns = style_patterns.get(expected_style, style_patterns['professional'])

        # Check for style-appropriate keywords
        keyword_count = sum(1 for kw in patterns.get('keywords', []) if kw in bullets_text)
        if keyword_count == 0:
            score -= 15

        # Check for metrics in executive style
        if expected_style == 'executive':
            has_metrics = bool(re.search(patterns.get('metrics', ''), bullets_text))
            if not has_metrics:
                score -= 20

        # Check for active voice (heuristic: avoid "is", "was", "been")
        passive_indicators = bullets_text.count(' is ') + bullets_text.count(' was ') + bullets_text.count(' been ')
        if passive_indicators > len(bullets):
            score -= 10  # Likely passive voice

        # Check for consistent verb tense
        present_tense = sum(1 for b in bullets if any(v in b.lower() for v in ['provides', 'enables', 'improves']))
        past_tense = sum(1 for b in bullets if any(v in b.lower() for v in ['provided', 'enabled', 'improved']))

        if present_tense > 0 and past_tense > 0 and len(bullets) > 2:
            score -= 10  # Mixed tenses

        return max(0, min(100, score))

    def _score_readability(self, bullets: List[str]) -> float:
        """
        Score readability (complexity, clarity)
        """
        score = 100.0

        # Combine bullets for readability analysis
        text = '. '.join(bullets)

        try:
            # Flesch Reading Ease (60-80 is ideal for slides)
            flesch = textstat.flesch_reading_ease(text)

            if flesch < 40:
                score -= 20  # Too hard
            elif flesch < 60:
                score -= 10
            elif flesch > 90:
                score -= 5  # Too simple
        except:
            pass  # If calculation fails, don't penalize

        # Average sentence complexity
        for bullet in bullets:
            words = bullet.split()

            # Penalize very long sentences
            if len(words) > 20:
                score -= 5

            # Penalize very short sentences (likely incomplete)
            if len(words) < 5:
                score -= 5

            # Check for overly complex words (>3 syllables)
            try:
                syllable_count = sum(textstat.syllable_count(word) for word in words)
                avg_syllables = syllable_count / len(words) if words else 0

                if avg_syllables > 2.5:
                    score -= 5  # Too complex
            except:
                pass

        return max(0, min(100, score))

    def _compare_to_golden(self, generated: List[str], expected: List[str]) -> float:
        """
        Compare to golden reference bullets (simple similarity measure)
        """
        if not expected:
            return None

        # Simple word-overlap similarity
        def jaccard_similarity(str1: str, str2: str) -> float:
            words1 = set(re.findall(r'\b\w+\b', str1.lower()))
            words2 = set(re.findall(r'\b\w+\b', str2.lower()))

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        # Find best match for each expected bullet
        similarities = []
        for exp_bullet in expected:
            best_sim = max(jaccard_similarity(exp_bullet, gen_bullet) for gen_bullet in generated)
            similarities.append(best_sim)

        # Average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Convert to 0-100 score
        return avg_similarity * 100

    def _check_conversational_filler(self, bullets: List[str], filler_list: List[str]) -> float:
        """
        Check for conversational filler phrases (returns penalty score 0-50)

        Used for transcript extraction quality - detects phrases like:
        - "As you've seen"
        - "I'd like to"
        - "Now let's"

        Returns penalty points (0 = no filler, higher = more filler found)
        """
        total_penalty = 0.0

        for bullet in bullets:
            bullet_lower = bullet.lower()

            # Check for each filler phrase
            for filler_phrase in filler_list:
                if filler_phrase.lower() in bullet_lower:
                    # Heavy penalty for conversational filler in extracted bullets
                    total_penalty += 20  # -20 points per occurrence

        return min(50, total_penalty)  # Cap at 50 point penalty

    def _check_sentence_truncation(self, bullets: List[str]) -> float:
        """
        Check for mid-sentence truncation (returns penalty score 0-40)

        Detects incomplete sentences that end abruptly like:
        - "...when it comes to applying."
        - "...whether that's."

        Returns penalty points (0 = no truncation, higher = truncation detected)
        """
        total_penalty = 0.0

        # Common truncation indicators
        truncation_patterns = [
            r'\b(when|where|what|whether|while|which|that)\s*["\']?s?\.',  # Ends with subordinating conjunction
            r'\b(to|of|for|with|by|in|on|at)\s*["\']?\.',  # Ends with preposition
            r'\b(and|or|but)\s*["\']?\.',  # Ends with coordinating conjunction
            r',\s*["\']?$',  # Ends with comma (incomplete)
        ]

        for bullet in bullets:
            bullet_stripped = bullet.strip()

            # Check each truncation pattern
            for pattern in truncation_patterns:
                if re.search(pattern, bullet_stripped, re.IGNORECASE):
                    total_penalty += 15  # -15 points per truncation
                    break  # Only penalize once per bullet

        return min(40, total_penalty)  # Cap at 40 point penalty

    def _check_sentence_completeness(self, bullets: List[str]) -> float:
        """
        Check that bullets are complete, well-formed sentences (returns penalty 0-30)

        Checks:
        - Minimum length (too short = likely incomplete)
        - Proper sentence structure
        - Not just fragments

        Returns penalty points (0 = complete sentences, higher = incomplete)
        """
        total_penalty = 0.0

        for bullet in bullets:
            words = bullet.strip().split()

            # Too short to be a complete sentence
            if len(words) < 5:
                total_penalty += 10
                continue

            # Check for sentence fragments (starts with conjunctions/pronouns only)
            first_word = words[0].lower()
            if first_word in ['and', 'or', 'but', 'because', 'since', 'while', 'although']:
                total_penalty += 5

            # Ends with ellipsis or incomplete punctuation
            if bullet.strip().endswith(('...', '..', '..')):
                total_penalty += 10

        return min(30, total_penalty)  # Cap at 30 point penalty

    def _empty_result(self) -> dict:
        """Return empty/failed result"""
        return {
            'structure_score': 0.0,
            'relevance_score': 0.0,
            'style_score': 0.0,
            'readability_score': 0.0,
            'overall_quality': 0.0,
            'bullet_count': 0,
            'avg_word_count': 0.0,
            'avg_char_length': 0.0
        }


def format_metrics_report(metrics: dict, test_id: str = None) -> str:
    """
    Format metrics as human-readable report
    """
    report = []

    if test_id:
        report.append(f"Test: {test_id}")
        report.append("=" * 60)

    report.append(f"Overall Quality: {metrics['overall_quality']:.1f}/100")
    report.append(f"  - Structure:    {metrics['structure_score']:.1f}/100")
    report.append(f"  - Relevance:    {metrics['relevance_score']:.1f}/100")
    report.append(f"  - Style:        {metrics['style_score']:.1f}/100")
    report.append(f"  - Readability:  {metrics['readability_score']:.1f}/100")

    if 'golden_similarity' in metrics and metrics['golden_similarity'] is not None:
        report.append(f"  - Golden Match: {metrics['golden_similarity']:.1f}/100")

    report.append(f"\nBullets: {metrics['bullet_count']}")
    report.append(f"Avg Length: {metrics['avg_word_count']:.1f} words ({metrics['avg_char_length']:.1f} chars)")

    return '\n'.join(report)
