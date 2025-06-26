import pytest
from src.agents.planning.content_analyzer import ContentAnalyzer

class TestContentAnalyzerRobust:
    def setup_method(self):
        self.analyzer = ContentAnalyzer()

    def test_empty_string(self):
        analysis = self.analyzer.analyze_content("")
        assert analysis['word_count'] == 0
        assert analysis['themes'] == []
        assert analysis['key_topics'] == []

    def test_long_content(self):
        content = "AI " * 10000 + "OpenAI released GPT-4."
        analysis = self.analyzer.analyze_content(content)
        assert analysis['word_count'] > 10000
        assert 'openai' in ''.join(analysis['companies']).lower() or 'gpt-4' in ''.join(analysis['models']).lower()

    def test_misspelled_terms(self):
        content = "Machin learnin algoritms and deep learing moduls."
        analysis = self.analyzer.analyze_content(content)
        # Should not crash, but may not match themes
        assert isinstance(analysis['themes'], list)

    def test_mixed_case_and_punctuation(self):
        content = "OpenAI, gPt-4! Google: Gemini-Ultra?"
        analysis = self.analyzer.analyze_content(content)
        assert any('openai' in c.lower() for c in analysis['companies'])
        assert any('gpt-4' in m.lower() for m in analysis['models'])
        assert any('gemini' in m.lower() or 'ultra' in m.lower() for m in analysis['models'])

    def test_ambiguous_terms(self):
        content = "Apple released a new model."
        analysis = self.analyzer.analyze_content(content)
        # Should recognize Apple as a company, but not as a fruit
        assert any('apple' in c.lower() for c in analysis['companies'])

    def test_non_english_content(self):
        content = "OpenAI ha lanzado GPT-4, un modelo de lenguaje avanzado."
        analysis = self.analyzer.analyze_content(content)
        # Should still extract some entities if present
        assert 'gpt-4' in ''.join(analysis['models']).lower() or 'openai' in ''.join(analysis['companies']).lower()

    def test_real_world_data_sample(self):
        content = (
            "Anthropic's Claude 3.5 Sonnet achieves new benchmarks in mathematical problem solving. "
            "Meta's Llama 3.1 introduces advanced reasoning and planning capabilities. "
            "NVIDIA's Blackwell B200 GPU architecture revolutionizes AI training and inference."
        )
        analysis = self.analyzer.analyze_content(content)
        assert any('anthropic' in c.lower() for c in analysis['companies'])
        assert any('claude' in m.lower() for m in analysis['models'])
        assert any('meta' in c.lower() for c in analysis['companies'])
        assert any('llama' in m.lower() for m in analysis['models'])
        assert any('nvidia' in c.lower() for c in analysis['companies'])
        assert any('blackwell' in m.lower() for m in analysis['models']) 