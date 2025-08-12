# Section-Aware Newsletter Generation Test Framework

Comprehensive test suite for the Phase 1 section-aware newsletter generation system, covering all components built as part of the multi-agent enhancement PRD.

## Overview

This test framework validates the four core section-aware components:

1. **Section-Aware Prompt Engine** (`section_aware_prompts.py`)
2. **Multi-Pass Section Processing** (`section_aware_refinement.py`)
3. **Section-Level Quality Metrics** (`section_quality_metrics.py`)
4. **Enhanced Continuity Validation** (`continuity_validator.py`)

## Test Structure

```
tests/
├── test_section_aware_prompts.py      # Prompt engine tests
├── test_section_aware_refinement.py   # Refinement system tests
├── test_section_quality_metrics.py    # Quality metrics tests
├── test_continuity_validator.py       # Continuity validation tests
├── test_integration.py                # Integration & workflow tests
├── conftest.py                        # Shared fixtures
├── run_section_aware_tests.py         # Test runner script
└── README_SECTION_AWARE_TESTS.md      # This file
```

## Quick Start

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov coverage
```

### Running Tests

**Run all tests with coverage:**
```bash
python tests/run_section_aware_tests.py --verbose
```

**Run only unit tests:**
```bash
python tests/run_section_aware_tests.py --unit-only
```

**Run with performance tests:**
```bash
python tests/run_section_aware_tests.py --include-slow
```

**Manual pytest execution:**
```bash
# From project root
pytest tests/test_section_aware_*.py -v --cov=src/core
```

## Test Categories

### Unit Tests

Each component has comprehensive unit tests covering:

#### Section-Aware Prompts (`test_section_aware_prompts.py`)
- ✅ Section type detection and validation
- ✅ Prompt generation for all section types
- ✅ Context handling and audience customization
- ✅ Template inheritance and customization
- ✅ Backward compatibility with existing systems
- ✅ Error handling and fallback behavior

#### Section Refinement (`test_section_aware_refinement.py`)
- ✅ Section boundary detection accuracy
- ✅ Multi-pass refinement logic
- ✅ Quality improvement tracking
- ✅ Content reassembly and flow preservation
- ✅ Section-specific improvement strategies
- ✅ Edge case handling

#### Quality Metrics (`test_section_quality_metrics.py`)
- ✅ Individual quality dimension scoring
- ✅ Section-specific quality weights
- ✅ Aggregated quality reporting
- ✅ Threshold validation and compliance
- ✅ Improvement recommendation generation
- ✅ Performance characteristics

#### Continuity Validation (`test_continuity_validator.py`)
- ✅ Transition analysis between sections
- ✅ Style consistency detection
- ✅ Content redundancy identification
- ✅ Issue severity classification
- ✅ Recommendation prioritization
- ✅ Multi-section coherence validation

### Integration Tests

Comprehensive workflow testing in `test_integration.py`:

- ✅ **End-to-End Processing**: Complete newsletter generation workflow
- ✅ **Component Interaction**: Data flow between all components
- ✅ **Quality Validation**: Integrated quality assessment
- ✅ **Performance Testing**: Scalability with large content
- ✅ **Error Handling**: Graceful degradation scenarios
- ✅ **Backward Compatibility**: Integration with existing systems

### Performance Tests

Benchmark testing for:
- Large content processing (>10K words)
- Multiple section types handling
- Concurrent operation performance
- Memory usage optimization
- Response time validation

## Test Coverage

Target coverage levels:
- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% workflow coverage  
- **Critical Paths**: 100% coverage for core functionality

View coverage reports:
```bash
# Generate and view HTML coverage report
python tests/run_section_aware_tests.py --verbose
open tests/coverage_html/index.html
```

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

**Content Fixtures:**
- `section_aware_newsletter_content`: Comprehensive sample newsletter
- `high_quality_section_content`: Well-structured content for testing
- `poor_quality_section_content`: Low-quality content for validation

**Context Fixtures:**
- `section_aware_context`: Standard generation context
- `comprehensive_context`: Full context with all parameters
- `minimal_context`: Minimal context for fallback testing

**Utility Functions:**
- `assert_valid_quality_score()`: Validate score ranges
- `assert_valid_content_length()`: Content validation
- `assert_valid_section_type()`: Section type validation

### Test Content Standards

**High-Quality Content Characteristics:**
- Clear section headers and structure
- Appropriate length for section type
- Engaging and informative language
- Proper transitions between sections
- Technical accuracy and citations

**Low-Quality Content Characteristics:**
- Minimal or unclear structure
- Inappropriate length
- Poor grammar and clarity
- Abrupt section transitions
- Missing technical details

## Running Specific Tests

### By Component
```bash
# Test prompt engine only
pytest tests/test_section_aware_prompts.py -v

# Test quality metrics only  
pytest tests/test_section_quality_metrics.py -v

# Test continuity validation only
pytest tests/test_continuity_validator.py -v
```

### By Test Type
```bash
# Integration tests only
pytest tests/test_integration.py -m integration -v

# Performance tests only
pytest tests/test_integration.py -m performance -v

# Exclude slow tests
pytest tests/test_section_aware_*.py -m "not slow" -v
```

### By Functionality
```bash
# Test specific functionality
pytest tests/ -k "test_section_detection" -v
pytest tests/ -k "test_quality_validation" -v
pytest tests/ -k "test_continuity" -v
```

## Continuous Integration

### GitHub Actions Integration

Add to `.github/workflows/test-section-aware.yml`:

```yaml
name: Section-Aware Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov coverage
    - name: Run section-aware tests
      run: python tests/run_section_aware_tests.py --verbose
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: tests/coverage.json
```

### Quality Gates

Tests include quality gates for:
- **Code Coverage**: >90% for critical components
- **Performance**: Sub-second response times for typical content
- **Quality Scores**: >0.8 for generated content
- **Error Rates**: <1% failure rate across test scenarios

## Troubleshooting

### Common Issues

**ImportError: Cannot import section-aware modules**
```bash
# Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Coverage not generated**
```bash
# Install coverage dependencies
pip install pytest-cov coverage
```

**Tests timeout on large content**
```bash
# Skip slow tests
pytest tests/ -m "not slow" -v
```

**Missing test fixtures**
```bash
# Verify conftest.py is in tests directory
ls tests/conftest.py
```

### Debug Mode

Enable detailed logging:
```bash
# Set log level for debugging
LOG_LEVEL=DEBUG python tests/run_section_aware_tests.py --verbose
```

View test output:
```bash
# Capture all output
pytest tests/test_section_aware_*.py -v -s --tb=long
```

## Performance Benchmarks

Expected performance characteristics:

| Operation | Content Size | Target Time | Max Memory |
|-----------|-------------|-------------|------------|
| Section Detection | 5K words | <100ms | <50MB |
| Quality Analysis | 3K words | <500ms | <100MB |
| Continuity Validation | 4 sections | <200ms | <75MB |
| Complete Workflow | 5K words | <2s | <200MB |

## Contributing to Tests

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_*.py` file
2. **Integration Tests**: Add to `test_integration.py`
3. **Fixtures**: Add shared data to `conftest.py`
4. **Performance**: Mark with `@pytest.mark.performance`

### Test Writing Guidelines

- Use descriptive test names: `test_quality_analysis_with_high_quality_content`
- Include docstrings explaining test purpose
- Use appropriate fixtures for test data
- Validate both success and failure scenarios
- Include edge cases and boundary conditions

### Example Test Structure

```python
def test_component_functionality(fixture_name):
    """Test component handles expected input correctly."""
    # Arrange
    component = ComponentClass()
    test_input = fixture_name
    
    # Act
    result = component.process(test_input)
    
    # Assert
    assert isinstance(result, ExpectedType)
    assert result.meets_requirements()
    assert_valid_quality_score(result.score)
```

## Test Results and Reporting

### Automated Reports

Test runner generates:
- **JSON Report**: `tests/section_aware_test_results.json`
- **HTML Coverage**: `tests/coverage_html/index.html`
- **Console Summary**: Real-time progress and results

### Report Contents

**Test Results:**
- Pass/fail status for each test suite
- Execution times and performance metrics
- Coverage percentages by component
- Error details and stack traces

**Quality Metrics:**
- Code coverage by file and function
- Test execution performance
- Component integration success rates
- Quality score distributions

## Next Steps

### Phase 2 Integration

This test framework will be extended for Phase 2 components:
- Sub-agent coordination testing
- Multi-agent workflow validation
- Advanced quality assessment integration
- Performance optimization validation

### Test Automation

Planned enhancements:
- Automated performance regression detection
- Quality trend analysis
- Continuous benchmark tracking
- Integration with deployment pipelines

---

For questions or issues with the test framework, refer to the main project documentation or create an issue in the project repository.