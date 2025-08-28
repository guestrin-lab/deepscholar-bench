# nuggetizer

[![PyPI](https://img.shields.io/pypi/v/nuggetizer?color=brightgreen)](https://pypi.org/project/nuggetizer/)
[![Downloads](https://static.pepy.tech/personalized-badge/nuggetizer?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nuggetizer)
[![Downloads](https://static.pepy.tech/personalized-badge/nuggetizer?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/nuggetizer)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![paper](https://img.shields.io/badge/paper-arxiv-blue.svg?style=flat)](https://arxiv.org/abs/2411.09607)

A powerful tool for information nugget creation, scoring, and assigning to RAG answers using LLMs.
Enables the evaluation of fact recall of RAG answers.

## 📟 Installation

### Create Conda Environment

```bash
conda create -n nuggetizer python=3.10
conda activate nuggetizer
```

### Pip Installation

```bash
pip install nuggetizer
```

### Development Installation

For development or the latest features, install from source:

```bash
git clone https://github.com/castorini/nuggetizer.git
cd nuggetizer
pip install -e .
```

### Environment Setup

Create a `.env` file with your OpenAI credentials. For Azure OpenAI (default for GPT models):

```bash
AZURE_OPENAI_API_BASE=your_azure_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_API_KEY=your_api_key
```

Or for OpenAI API:

```bash
OPEN_AI_API_KEY=your_openai_api_key
```

## 🚀 Quick Start

Here's a simple example of how to use nuggetizer:

```python
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer

# Create a sample request
query = Query(qid="1", text="What are the main features of Python?")
documents = [
    Document(
        docid="1",
        segment="""Python is a high-level programming language known for its 
        simplicity and readability. It supports multiple programming paradigms 
        including procedural, object-oriented, and functional programming."""
    ),
    Document(
        docid="2",
        segment="""Python was created by Guido van Rossum in 1991."""
    ),
    Document(
        docid="3",
        segment="""Python is widely used in web development, data analysis, 
        artificial intelligence, and scientific computing."""
    ),
]
request = Request(query=query, documents=documents)

# Option 1: Single model for all components
nuggetizer = Nuggetizer(model="gpt-4o")  # Uses same model for all components

# Option 2: Different models for each component
nuggetizer_mixed = Nuggetizer(
    creator_model="gpt-4o",  # Model for nugget creation
    scorer_model="gpt-3.5-turbo",  # Model for nugget scoring
    assigner_model="gpt-4o"  # Model for nugget assignment
)

# Create and score nuggets
scored_nuggets = nuggetizer.create(request)

# Print nuggets and their importance
for nugget in scored_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}\n")

# Assign nuggets to a specific document
assigned_nuggets = nuggetizer.assign(query.text, documents[0].segment, scored_nuggets)

# Print assignments
for nugget in assigned_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}")
    print(f"Assignment: {nugget.assignment}\n")
```

You can also run a little more elaborate example with:
```bash
python3 examples/e2e.py
```

We also provide an async version of the Nuggetizer class, `AsyncNuggetizer`, in `src/nuggetizer/models/async_nuggetizer.py`. To run this example, use:

```bash
python3 examples/async_e2e.py
```

## 🛠️ Components

The Nuggetizer class provides a unified interface for:

1. **Nugget Creation & Scoring**: Extracts and scores atomic information nuggets from text
2. **Nugget Assignment**: Assigns nuggets to specific texts

The following scripts are provided to help you with through the process for the TREC 2024 RAG Track:

1. First, generate nuggets:
```bash
# Extract nuggets
python3 scripts/create_nuggets.py --input_file pool.jsonl --output_file nuggets.jsonl --log_level 1
```

2. For RAG answers, we assume they take on the format laid out by the wonderful [TREC 2024 RAG Track](https://trec-rag.github.io/annoucements/2024-track-guidelines/):

```json
{
    "run_id": "ragnarok",
    "topic_id": "2027497",
    "topic": "how often should you take your toddler to the potty when potty training",
    "references": [
        "msmarco_v2.1_doc_51_766815931#2_1606878413", "msmarco_v2.1_doc_51_766815931#1_1606876582", "msmarco_v2.1_doc_51_766815931#5_1606882767", 
        "msmarco_v2.1_doc_51_766815931#6_1606884302", "msmarco_v2.1_doc_51_766815931#3_1606879951", "msmarco_v2.1_doc_51_766815931#4_1606881348", 
        "msmarco_v2.1_doc_37_463237391#10_984448281", "msmarco_v2.1_doc_51_766815931#0_1606874600", "msmarco_v2.1_doc_37_463237391#9_984446615", 
        "msmarco_v2.1_doc_28_472446307#22_1012988885", "msmarco_v2.1_doc_51_766815931#7_1606885873", "msmarco_v2.1_doc_28_472446307#21_1012986800", 
        "msmarco_v2.1_doc_29_562342450#23_1356565296", "msmarco_v2.1_doc_29_562342450#17_1356555947", "msmarco_v2.1_doc_49_418787959#7_861728734", 
        "msmarco_v2.1_doc_49_418787959#6_861726964", "msmarco_v2.1_doc_26_680625866#7_1289507527", "msmarco_v2.1_doc_10_1346272776#19_2165266355", 
        "msmarco_v2.1_doc_56_1491300640#3_3012150696", "msmarco_v2.1_doc_10_672519892#5_1260010758"], 
    "response_length": 192, 
    "answer": [
            {"text": "The frequency with which you should take your toddler to the potty depends on their readiness for potty training.", "citations": [0, 1, 12, 13, 19]}, 
            {"text": "Some sources suggest that toddlers should be taken to the potty about three times a day: first thing in the morning, after mealtimes, and again before bedtime.", "citations": [0, 4, 6, 8]}, 
            {"text": "It is recommended that you watch for facial expressions or poses that may signal that they need to \"go\".", "citations": [6, 8]}, 
            {"text": "If they are reluctant to use the potty, don't force them.", "citations": [6, 8]}, 
            {"text": "Other sources suggest that toddlers should be taken to the potty every two hours, whether they have to go or not.", "citations": [14, 15]}, 
            {"text": "This includes first thing in the morning, before leaving the house, and before naps and bedtime.", "citations": [14, 15]}, 
            {"text": "Some sources recommend taking toddlers to the potty every 30 minutes to an hour.", "citations": [9, 11, 17]}, 
            {"text": "This is to increase the chances of them peeing in the potty instead of on the floor.", "citations": [9, 11]}, 
            {"text": "It is important to keep in mind that every toddler is different, and their potty training journey will be unique to them.", "citations": [0, 4]}, 
            {"text": "It is recommended that you let your toddler lead the way and be gentle throughout the process, as their self-esteem can be fragile during this time.", "citations": [0, 1]}
        ]
}
```
To *easily* generate answers in this format, consider using [Ragnarök](https://github.com/castorini/ragnarok).
Let's now assign the nuggets to the RAG answers:

```bash
# Assign nuggets to RAG answers
python3 scripts/assign_nuggets.py \
    --nugget_file nuggets.jsonl \
    --answer_file ragnarok.jsonl \
    --output_file final_assignments.jsonl

# Calculate metrics
python3 scripts/calculate_metrics.py \
    --input_file final_assignments.jsonl \
    --output_file metrics.jsonl
```

The final output file (`final_assignments.jsonl`) will contain:
- query: The original query
- qid: Query ID
- answer_text: Full answer text
- response_length: Response length
- run_id: Run ID (derived from the RAG answer filename)
- nuggets: Nuggets with their importance labels and assignments

The final metrics file (`metrics.jsonl`) will contain:
- Per-response metrics:
  - `strict_vital_score`: Score counting only full support for vital nuggets
  - `strict_all_score`: Score counting only full support for all nuggets
  - `vital_score`: Score counting full (1.0) and partial (0.5) support for vital nuggets
  - `all_score`: Score counting full (1.0) and partial (0.5) support for all nuggets
- Global mean metrics across all responses (indicated by `qid` as `all`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is built with the support of Azure's OpenAI credits.

## ✨ References

If you use Nuggetizer, please cite the following relevant papers:

[[2411.09607] Initial Nugget Evaluation Results for the {TREC 2024 RAG Track} with the {AutoNuggetizer Framework}](https://arxiv.org/abs/2411.09607)

```
@ARTICLE{pradeep2024autonuggetizer,
  title   = {Initial Nugget Evaluation Results for the {TREC 2024 RAG Track} with the {AutoNuggetizer Framework}},
  author  = {Ronak Pradeep and Nandan Thakur and Shivani Upadhyay and Daniel Campos and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2411.09607}
}
```

[[2406.16828] Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track](https://arxiv.org/abs/2406.16828)
```
@ARTICLE{pradeep2024ragnarok,
  title   = {{Ragnarök}: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track},
  author  = {Ronak Pradeep and Nandan Thakur and Sahel Sharifymoghaddam and Eric Zhang and Ryan Nguyen and Daniel Campos and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2406.16828},
}
```
