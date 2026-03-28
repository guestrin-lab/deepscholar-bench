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
8. **Quality filtering papers** based on related works completeness and citation counts
9. **Adding reference citation counts** using OpenAlex API for all references in related works

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
    --citation_input_file all_citations.csv \
    --related_works_input_file paper_content_filtered_with_citations.csv \
    --model gpt-4o \
    --output_file important_citations.csv
```

**What it does:**
- Analyzes each citation in context of its parent paper (title, abstract, related works section)
- Determines if citations are important vs. tangential/substitutable
- Filters out non-important references
- Provides a curated list of the most important prior work

**Prerequisites:**
- Combined citations CSV file (if you have individual citation files, combine them first)
- Papers CSV file with related works sections

**Configuration Options:**
- `--citation_input_file`: Combined citations CSV file (required)
- `--related_works_input_file`: Papers CSV file with related works sections (required)
- `--model`: LLM model to use (required, e.g., gpt-4o)
- `--output_file`: Output important citations CSV file (required)

**Note:** If you have individual citation CSV files (one per paper), you'll need to combine them first before running this script. You can use a simple script or pandas to concatenate all citation files from the `outputs/citations/` folder.

### Step 4: Nugget Generation (`generate_nuggets_from_reports.py`)

Extracts key informational "nuggets" from related works sections using the nuggetizer library.

```bash



# Specify input directory (default: outputs/20251015_143311)
python data_pipeline/generate_nuggets_from_reports.py \
    --input_dir outputs/20251015_143311

# Use different CSV file and model
python data_pipeline/generate_nuggets_from_reports.py \
    --input_dir outputs/20251015_143311 \
    --csv_file paper_content_filtered.csv \
    --model gpt-4o \

```

**What it does:**
- Processes related works sections from papers to identify key informational nuggets
- Uses LLM to extract atomic information claims from related works sections
- Scores nuggets by importance (vital vs. okay)


**Output Structure:**
- Creates `nuggets/` folder in the input directory (or specified output directory)
- Each paper gets its own folder: `nuggets/{arxiv_id}/res.json`
- Output JSON contains:
  - `qid`: ArXiv ID
  - `query`: The prompt used (related works section generation based on abstract)
  - `nuggets`: All extracted nuggets with importance and assignment
  - `supported_nuggets`: Fully supported nuggets only
  - `partially_supported_nuggets`: Nuggets with partial or full support

**Configuration Options:**
- `--input_dir`: Path to input directory containing CSV file (default: outputs/20251015_143311)
- `--csv_file`: CSV file name in input directory (default: paper_content_filtered.csv)
- `--output_dir`: Output directory for nuggets (default: input_dir/nuggets)
- `--model`: LLM model for nuggetizer (default: gpt-4o)
- `--log_level`: Logging verbosity 0-2 (default: 1)
- `--skip_existing`: Skip papers that already have nuggets generated
- `--use_azure_openai`: Use Azure OpenAI instead of standard OpenAI
- `--use_vllm`: Use vLLM API endpoint

### Step 5: Quality Filtering (`filter_quality_papers.py`)

Filters papers based on related works quality and citation count to ensure high-quality datasets for downstream tasks.

```bash
# Single folder - standard quality
python filter_quality_papers.py --input-folder outputs/20251015_143311/

# All folders in outputs/
./batch_filter_papers.sh

# Custom thresholds
python filter_quality_papers.py \
    --input-folder outputs/20251015_143311/ \
    --min-citations 10 \
    --min-rw-length 500 \
    --max-rw-length 8000
```

**What it does:**
- Filters papers with no related works section
- Removes papers with very short related works (< 200 chars default)
- Removes papers with excessively long related works (> 10000 chars default)
- Filters papers with fewer than 5 citations (default)

**Quality Presets:**

| Goal | min_citations | min_rw_length | max_rw_length |
|------|--------------|---------------|---------------|
| **Strict** | 15 | 800 | 8000 |
| **Standard** | 5 | 200 | 10000 |
| **Lenient** | 3 | 100 | 15000 |
| **Surveys** | 20 | 1000 | 20000 |

**Outputs:**
- New folder with the name of the input folder, but with the suffix _filtered or the name of the output folder specified in the command.
    - `paper_content.csv` - Filtered paper data
    - `papers.csv` - Filtered paper metadata
    - `papers_with_related_works.csv` - Filtered paper and related works data
    - `related_works_combined.csv` - Filtered related works data
    - `citations.csv` - Filtered citations data

## 📊 Output Files

### Core Pipeline Outputs (Step 1)

| File | Description | Key Columns |
|------|-------------|-------------|
| `papers_*.csv` | Raw ArXiv paper metadata | `arxiv_id`, `title`, `authors`, `abstract`, `categories` |
| `paper_content_*.csv` | Papers with extracted related works | `paper_title`, `related_works_section`, `related_works_length` |
| `citations_*.csv` | Individual citations from papers | `parent_paper_title`, `cited_paper_title`, `has_metadata`, `is_arxiv_paper` |
| `citation_stats_*.csv` | Aggregated citation statistics | `total_citations`, `resolution_rate`, `arxiv_citation_rate` |
| `papers_with_related_works_*.csv` | **Main output**: Combined paper and related works data | All paper metadata + related works content |

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

**Important Citations:**
- `--citation_input_file`: Combined citations CSV file (required)
- `--related_works_input_file`: Papers CSV file with related works sections (required)
- `--model`: LLM model (e.g., gpt-4o, required)
- `--output_file`: Output important citations CSV file (required)

**Nugget Generation:**
- `--input_dir`: Path to input directory containing CSV file (default: outputs/20251015_143311)
- `--csv_file`: CSV file name in input directory (default: paper_content_filtered.csv)
- `--output_dir`: Output directory for nugget files (default: input_dir/nuggets)
- `--model`: LLM model (e.g., gpt-4o, default: gpt-4o)
- `--log_level`: Logging verbosity (0=warning, 1=info, 2=debug, default: 1)
- `--skip_existing`: Skip papers that already have nuggets generated
- `--use_azure_openai`: Use Azure OpenAI instead of standard OpenAI
- `--use_vllm`: Use vLLM API endpoint

**Quality Filtering:**
- `--input-folder`: Path to folder containing `paper_content.csv` (required)
- `--min-citations`: Minimum number of citations required (default: 5)
- `--min-rw-length`: Minimum related works length in characters (default: 200)
- `--max-rw-length`: Maximum related works length in characters (default: 10000)
- `--citations-folder`: Path to citations folder (default: ../citations/)
- `--output-suffix`: Suffix for output filenames (default: _filtered)

**Reference Citation Counts:**
- `--papers-csv`: Path to papers CSV file (required)
- `--citations-folder`: Path to folder containing citation CSV files (required)
- `--output-csv`: Output CSV path (if not provided, overwrites input)
- `--rate-limit-delay`: Delay between API requests in seconds (default: 0.05)

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
    --citation_input_file outputs/20240101_120000/all_citations.csv \
    --related_works_input_file outputs/20240101_120000/paper_content_filtered_with_citations.csv \
    --model gpt-4o \
    --output_file outputs/20240101_120000/important_citations.csv

# Step 4: Generate nuggets from related works
python -m data_pipeline.generate_nuggets_from_reports \
    --input_dir outputs/20240101_120000 \
    --model gpt-4o

# Step 5: Filter for quality papers
python -m data_pipeline.filter_quality_papers --input-folder outputs/20240101_120000/ --output-folder outputs/20240101_120000_filtered/

# Step 6: Add reference citation counts
python -m data_pipeline.add_reference_citations \
    --papers-csv outputs/20240101_120000_filtered/paper_content.csv \
    --citations-path outputs/20240101_120000_filtered/citations.csv \
    --output-csv outputs/20240101_120000_filtered/paper_content_with_citations.csv
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
- **Post-processing filtering**: Removes papers with incomplete/inadequate related works sections
- **Citation quality**: Ensures papers have sufficient extracted citations (configurable threshold)
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
- **OpenAlex API**: Free API for citation counts (respects rate limits with built-in delays)
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