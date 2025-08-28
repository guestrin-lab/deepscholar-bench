# ArXiv Data Collection Pipeline

A comprehensive pipeline for collecting ArXiv papers, extracting related works sections, and performing citation analysis with metadata lookup and nugget generation.

## 🎯 Overview

This pipeline automates the complete workflow from paper discovery to insight extraction:
1. **Scraping ArXiv papers** based on categories and date ranges
2. **Filtering papers** by author h-index criteria
3. **Extracting related works sections** from PDF files (clean text) and LaTeX source files (bibliography)
4. **Extracting and resolving citations** with metadata lookup
5. **Recovering missing citation metadata** using external APIs
6. **Identifying important citations** using LLM-based filtering
7. **Generating informational nuggets** from related works sections

## 🚀 Complete Workflow

### Step 1: Main Data Collection (`main.py`)

The primary script for collecting papers and their citations from ArXiv.

```bash
# Basic usage with default settings (last 30 days, cs.AI/cs.CL/cs.LG categories)
python -m data_pipeline.main

# Process a single paper by ArXiv ID
python -m data_pipeline.main --paper-id 2502.07374

# Custom run with specific categories and date range
python -m data_pipeline.main --categories cs.AI cs.CV --start-date 2024-01-01 --max-papers-per-category 20
```

**What it does:**
- Scrapes ArXiv papers based on search criteria
- Filters papers by author h-index (default: min 20)
- Downloads PDF and LaTeX source files
- Extracts related works sections from PDFs (clean text)
- Extracts citations from LaTeX bibliography files
- Resolves basic citation metadata using ArXiv API

**Key Outputs:**
- `papers_YYYYMMDD_HHMMSS.csv` - Raw paper metadata
- `citations_YYYYMMDD_HHMMSS.csv` - Extracted citations
- `papers_with_related_works_YYYYMMDD_HHMMSS.csv` - Combined paper and related works data

### Step 2: Citation Recovery (`recover_citations.py`)

Enhances citation data by recovering missing metadata using ArXiv and Tavily APIs.

```bash
python -m data_pipeline.recover_citations \
    --input_file citations_YYYYMMDD_HHMMSS.csv \
    --output_file recovered_citations.csv
```

**What it does:**
- Searches ArXiv API for exact title matches
- Falls back to Tavily search for non-ArXiv papers
- Adds missing titles, URLs, abstracts, and content snippets
- Significantly improves citation resolution rates

### Step 3: Essential Citation Filtering (`get_important_citations.py`)

Uses LLM analysis to identify citations that are truly important to each paper's related works.

```bash
python -m data_pipeline.get_important_citations \
    --citation_input_file recovered_citations.csv \
    --related_works_input_file papers_with_related_works_YYYYMMDD_HHMMSS.csv \
    --model gpt-4o \
    --output_file important_citations.csv
```

**What it does:**
- Analyzes each citation in context of its parent paper
- Determines if citations are important vs. tangential/substitutable
- Filters out non-important references
- Provides a curated list of the most important prior work

### Step 4: Nugget Generation (`generate_nuggets_from_reports.py`)

Extracts key informational "nuggets" from related works sections using the nuggetizer library.

```bash
python -m data_pipeline.generate_nuggets_from_reports \
    --output_dir gt_nuggets_outputs \
    --model gpt-4.1
```

**What it does:**
- Processes related works sections to identify key claims
- Scores nuggets by importance and relevance
- Assigns support levels (support, partial_support, etc.)
- Calculates comprehensive metrics for each paper
- Outputs structured JSON and CSV files with detailed nugget analysis

## 📊 Output Files

### Core Pipeline Outputs (Step 1)

| File | Description | Key Columns |
|------|-------------|-------------|
| `papers_*.csv` | Raw ArXiv paper metadata | `arxiv_id`, `title`, `authors`, `abstract`, `categories` |
| `paper_content_*.csv` | Papers with extracted related works | `paper_title`, `related_works_section`, `related_works_length` |
| `citations_*.csv` | Individual citations from papers | `parent_paper_title`, `cited_paper_title`, `has_metadata`, `is_arxiv_paper` |
| `citation_stats_*.csv` | Aggregated citation statistics | `total_citations`, `resolution_rate`, `arxiv_citation_rate` |
| `papers_with_related_works_*.csv` | **Main output**: Combined paper and related works data | All paper metadata + related works content |

### Enhanced Outputs (Steps 2-4)

| File | Description |
|------|-------------|
| `recovered_citations.csv` | Citations with enhanced metadata from external APIs |
| `important_citations.csv` | LLM-filtered list of important citations only |
| `gt_nuggets_outputs/` | Directory containing nugget analysis for each paper (JSON + CSV) |

## ⚙️ Configuration Options

### Main Pipeline (`main.py`)

**Data Collection:**
- `--categories [LIST]`: ArXiv categories (default: cs.AI cs.CL cs.LG)
- `--start-date YYYY-MM-DD`: Start date (default: 30 days ago)
- `--end-date YYYY-MM-DD`: End date (default: today)
- `--max-papers-per-category INT`: Papers per category (default: 50)

**Quality Filtering:**
- `--min-hindex INT`: Minimum author h-index (default: 20)
- `--max-hindex INT`: Maximum h-index upper bound
- `--min-citations INT`: Minimum citations in related works (default: 5)

**Processing:**
- `--paper-id ARXIV_ID`: Process single paper by ID
- `--output-dir PATH`: Output directory (default: data_pipeline/outputs)
- `--concurrent-requests INT`: API concurrency (default: 5)
- `--request-delay FLOAT`: API delay in seconds (default: 1.0)

### Downstream Scripts

**Citation Recovery:**
- `--input_file`: Input citations CSV
- `--output_file`: Output enhanced citations CSV

**Essential Citations:**
- `--citation_input_file`: Citations CSV (preferably recovered)
- `--related_works_input_file`: Papers with related works CSV
- `--model`: LLM model (e.g., gpt-4o)
- `--output_file`: Output important citations CSV

**Nugget Generation:**
- `--output_dir`: Output directory for nugget files
- `--model`: LLM model (e.g., gpt-4.1)
- `--log_level`: Logging verbosity (0=warning, 1=info, 2=debug)

## 📋 Example Workflows

### Complete End-to-End Pipeline

```bash
# Step 1: Collect papers and initial citations
python -m data_pipeline.main --categories cs.AI --max-papers-per-category 10

# Step 2: Enhance citation metadata
python -m data_pipeline.recover_citations \
    --input_file outputs/20240101_120000/citations.csv \
    --output_file recovered_citations.csv

# Step 3: Filter to important citations
python -m data_pipeline.get_important_citations \
    --citation_input_file recovered_citations.csv \
    --related_works_input_file outputs/20240101_120000/papers_with_related_works.csv \
    --model gpt-4o \
    --output_file important_citations.csv

# Step 4: Generate nuggets from related works
python -m data_pipeline.generate_nuggets_from_reports \
    --output_dir nugget_analysis \
    --model gpt-4.1
```

### Quick Testing

```bash
# Minimal test run
python -m data_pipeline.main \
    --categories cs.AI \
    --min-hindex 0 \
    --max-papers-per-category 3 \
    --start-date 2024-05-01
```

### Single Paper Analysis

```bash
# Analyze a specific paper
python -m data_pipeline.main --paper-id 2502.07374 --min-hindex 0
```

## 🔧 Technical Features

### Hybrid Processing Approach
- **PDF extraction**: Clean text without LaTeX artifacts for content analysis
- **LaTeX extraction**: Precise bibliography and citation parsing from source files
- **Multi-file handling**: Processes complete LaTeX projects with multiple .tex and .bib files

### Advanced Citation Resolution
- **Multi-strategy search**: ArXiv API → Tavily search → fuzzy matching
- **Semantic similarity**: Matches citations using title/author similarity
- **Metadata enrichment**: Adds abstracts, DOIs, journal references, publication dates

### Quality Control
- **Author filtering**: H-index-based quality filtering using Semantic Scholar
- **Content validation**: Minimum citation requirements and section length checks
- **Error handling**: Graceful failures with detailed logging
- **Rate limiting**: Configurable delays to respect API limits

## 🔍 Monitoring & Debugging

### Logging Levels
- `INFO`: Progress updates and summary statistics
- `DEBUG`: Detailed processing information  
- `WARNING`: Non-fatal issues (papers without related works)
- `ERROR`: Fatal processing errors

### Output Verification
1. Check CSV files are generated in output directory
2. Review log messages for processing steps
3. Verify summary statistics are reasonable
4. Sample data quality in generated files

## ⚠️ Important Considerations

### API Usage
- **ArXiv API**: Respect rate limits, use appropriate delays
- **Semantic Scholar**: Author h-index lookup may fail for some authors
- **Tavily API**: Requires API key for citation recovery
- **OpenAI/Anthropic**: Required for LLM-based important citation filtering and nugget generation

### Resource Requirements
- **Memory**: Large datasets require significant RAM for processing
- **Storage**: CSV files can become large with many papers
- **Time**: Complete pipeline can take hours for large date ranges
- **API Costs**: LLM-based steps (Steps 3-4) incur API usage costs

### Data Quality Notes
- Citation resolution rates vary (typically 60-80% after recovery)
- LaTeX parsing may fail for non-standard formatting
- H-index data availability varies by author
- Essential citation filtering quality depends on LLM model choice

## 📄 License

This project is part of the DeepScholarBench pipeline for academic paper analysis.