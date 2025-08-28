# 📊 Research Paper Evaluation Suite

A comprehensive evaluation framework for comparing the performance of various research paper processing and summarization systems. This suite provides standardized metrics and automated evaluation workflows to assess the quality, accuracy, and completeness of AI-generated research summaries.

## 🎯 Overview

The evaluation suite supports multiple research processing systems and provides detailed metrics for:
- Content organization and structure
- Information coverage and completeness  
- Citation accuracy and relevance
- Claim verification and support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required dependencies (see `requirements.txt` in project root)
- Access to evaluation datasets (located in `dataset/`)

### Basic Usage

Run the evaluation suite from the project root directory:

```bash
# Evaluate a single system with multiple metrics
python -m eval.main \
  --mode deepscholar_base \
  --evals nugget_coverage reference_coverage \
  --input_folder tests/baselines_results/deepscholar_base_gpt_4.1 \
  --output_folder evaluation_results/

# Evaluate multiple systems simultaneously
python -m eval.main \
  --mode deepscholar_base openscholar storm \
  --evals organization nugget_coverage reference_coverage \
  --input_folder tests/baselines_results/deepscholar_base_gpt_4.1 tests/baselines_results/openscholar tests/baselines_results/storm \
  --output_folder evaluation_results/

# Process specific papers only
python -m eval.main \
  --mode deepscholar_base \
  --evals nugget_coverage \
  --input_folder tests/baselines_results/deepscholar_base_gpt_4.1 \
  --file_id 0 1 2 \
  --output_folder evaluation_results/
```

## 📋 Configuration Options

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--modes` | Systems to evaluate | `deepscholar_base openscholar storm` |
| `--evals` | Evaluation metrics to run | `nugget_coverage reference_coverage` |
| `--input_folder` | Path to system outputs | `tests/baselines_results/` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_folder` | `results` | Evaluation results directory |
| `--file_id` | All files | Specific file IDs to process |
| `--model_name` | `gpt-4o` | LLM model for evaluation |
| `--dataset_path` | `test/dataset/related_works_combined.csv` | Dataset file path |
| `--essential_citations_path` | `test/dataset/essential_citations.csv` | Essential citations file |
| `--nugget_groundtruth_dir_path` | `test/dataset/nugget_groundtruth` | Ground truth nuggets directory |

## 🎯 Supported Systems

The evaluation suite supports the following research processing systems:

| System | Description | Parser Module |
|--------|-------------|---------------|
| **DeepScholar_Base** | AI-powered research summarization | `parsers/deepscholar_base.py` |
| **OpenScholar** | Academic paper analysis | `parsers/openscholar.py` |
| **DeepResearcher** | Deep learning research assistant | `parsers/deepresearcher.py` |
| **STORM** | Structured topic outline research | `parsers/storm.py` |
| **SearchAI** | Search-based AI research | `parsers/search_ai.py` |
| **GroundTruth** | Human-annotated baseline | `parsers/groundtruth.py` |

## 📊 Evaluation Metrics

### Content Quality Metrics

- **🏗️ Organization** (`organization`)
  - Evaluates logical structure, flow, and coherence
  - Assesses introduction, body, and conclusion quality
  - Scores: 1-5 scale for overall organization

- **💎 Nugget Coverage** (`nugget_coverage`) 
  - Measures coverage of key information pieces
  - Compares against human-annotated ground truth
  - Metrics: Coverage rate, precision, recall

### Citation & Reference Metrics

- **📚 Reference Coverage** (`reference_coverage`)
  - Proportion of ground truth references cited
  - Identifies missing critical citations
  - Metrics: Coverage percentage, citation overlap

- **🎯 Citation Precision** (`cite_p`)
  - Accuracy and relevance of citations
  - Measures proper citation formatting
  - Metrics: Precision, accuracy scores

- **📈 Document Importance** (`document_importance`)
  - Assesses importance of cited documents
  - Weights citations by paper impact
  - Metrics: Weighted importance scores

### Content Verification Metrics

- **✅ Claim Coverage** (`claim_coverage`)
  - Verification that claims are supported by sources
  - Identifies unsupported statements
  - Metrics: Support rate, verification confidence

- **🔍 Coverage Relevance Rate** (`coverage_relevance_rate`)
  - Relevance of covered information to the topic
  - Measures information quality vs quantity
  - Metrics: Relevance scores, signal-to-noise ratio

## 📁 Input Data Structure

The evaluation suite expects the following directory structure:

```
input_folder/
├── system_name/
│   ├── 0/                    # Paper ID folders
│   │   ├── output.md       # System output
│   │   └── metadata.json     # Optional metadata
│   ├── 1/
│   └── ...
└── another_system/
    └── ...
```

## 📤 Output Structure

Results are saved in the specified output folder:

```
output_folder/
├── results.csv              # Aggregated results across all metrics
├── organization/
│   ├── aggregated_results.csv
│   └── system_name.csv
├── nugget_coverage/
│   └── ...
```

## 🔧 Adding New Systems

To add support for a new research system:

1. Create a new parser in `eval/parsers/your_system.py`
2. Inherit from the base `Parser` class
3. Implement required methods for parsing system output
4. Add the system to `ParserType` enum in `parser_type.py`
5. Update the parser factory in `__init__.py`

Example parser structure:
```python
from .parser import Parser

class YourSystemParser(Parser):
    def parse_content(self) -> str:
        # Extract main content from system output
        pass
    
    def parse_citations(self) -> list:
        # Extract citation information
        pass
```

## 🔧 Adding New Metrics

To implement a new evaluation metric:

1. Create a new evaluator in `eval/evaluator/your_metric.py`
2. Inherit from the base evaluator class
3. Implement the `calculate()` and `aggregate()` methods
4. Add the metric to `EvaluationFunction` enum
5. Update the factory method

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Support

For questions, issues, or contributions:
- 🐛 [Report bugs](https://github.com/guestrin-lab/deepscholar-bench/issues)
- 💡 [Request features](https://github.com/guestrin-lab/deepscholar-bench/issues)
