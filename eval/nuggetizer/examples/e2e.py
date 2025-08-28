#!/usr/bin/env python3
import argparse
import time
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores


def create_sample_request() -> Request:
    """Create a sample request with a query and documents."""
    query = Query(
        qid="sample-1",
        text="What are the key benefits and features of Python programming language?",
    )

    documents = [
        Document(
            docid="doc1",
            segment="""Python is renowned for its simplicity and readability, making it an excellent choice for beginners. 
            Its extensive standard library provides built-in support for many programming tasks.""",
        ),
        Document(
            docid="doc2",
            segment="""Python supports multiple programming paradigms including object-oriented, imperative, and functional 
            programming. It has a large and active community that contributes to thousands of third-party packages.""",
        ),
        Document(
            docid="doc3",
            segment="""Python's dynamic typing and automatic memory management help developers focus on solving problems 
            rather than managing low-level details. It's widely used in web development, data science, and AI.""",
        ),
        Document(
            docid="doc4",
            segment="""Python is open source. And some features included are:
            - Lambda functions are anonymous functions that can be used to create quick, one-time-use functions.
            - List comprehensions are a way to create lists in a concise manner.
            - Generators are a way to create iterators in a concise manner.
            - Decorators are a way to modify or enhance functions or methods.
            - Context managers are a way to manage resources in a concise manner.
            - Async/await are a way to write asynchronous code in a concise manner.
            - Type hints are a way to add type information to variables and function parameters.
            - etc.
            """,
        ),
        Document(
            docid="doc5",
            segment="""Benefits of Python Programming Language

1. Simplicity & Readability
	•	Intuitive, easy-to-read syntax reduces development time.
	•	Indentation-based structure enforces clean, readable code.

2. Versatility & General-Purpose Use
	•	Supports multiple paradigms: procedural, object-oriented, functional.
	•	Used in web development, data science, machine learning, automation, scripting, etc.

3. Extensive Libraries & Ecosystem
	•	Rich standard library (e.g., json, os, sys, datetime).
	•	Strong third-party ecosystem: NumPy, Pandas (data science), TensorFlow, PyTorch (ML), Flask, Django (web), Scrapy (web scraping).

4. Cross-Platform Compatibility
	•	Runs on Windows, macOS, Linux, and embedded systems.
	•	Code can be executed in various environments without modification.

5. Strong Community & Support
	•	Large open-source community with extensive documentation and forums (Stack Overflow, GitHub).
	•	Frequent updates and improvements from contributors worldwide.

6. Rapid Development & Prototyping
	•	Dynamic typing and interpreted nature allow fast iteration.
	•	Ideal for quick prototyping before optimizing in lower-level languages.

7. Integration & Interoperability
	•	Can interface with C/C++ (ctypes, Cython), Java (Jython), .NET (IronPython).
	•	Supports API calls via HTTP, WebSockets, and external services.

8. Scalability & Maintainability
	•	Modular design and structured code make large projects manageable.
	•	Popular in enterprise applications (Dropbox, Instagram, Reddit).

9. High Demand & Career Opportunities
	•	Widely used in industry, academia, and research.
	•	High job demand in data science, AI, software development, cybersecurity, etc.

10. Strong for AI & Data Science
	•	Dominates ML & AI fields with TensorFlow, PyTorch, and Scikit-learn.
	•	Integrated with big data tools (Apache Spark, Dask).

11. Automation & Scripting
	•	Ideal for automating repetitive tasks (web scraping, testing, system administration).
	•	Popular in DevOps workflows (Ansible, Fabric).

12. Open-Source & Free
	•	No licensing costs, freely available for any use case.
	•	Thrives on an active open-source ecosystem.

Conclusion

Python's simplicity, versatility, vast ecosystem, and strong community support make it one of the most powerful and widely used programming languages across industries.""",
        ),
    ]

    return Request(query=query, documents=documents)


def process_request(
    request: Request, model: str, use_azure_openai: bool, use_vllm: bool, log_level: int
) -> None:
    """Process a request through the nuggetizer pipeline."""
    start_time = time.time()

    print("🚀 Initializing components...")
    # Initialize components - API keys and Azure config are loaded automatically

    # Option 1: Single model for all components
    nuggetizer1 = Nuggetizer(
        model=model,
        use_azure_openai=use_azure_openai,
        use_vllm=use_vllm,
        log_level=log_level,
    )

    # Option 2: Different models for each component
    # nuggetizer2 = Nuggetizer(
    #     creator_model="gpt-4o",
    #     scorer_model="gpt-3.5-turbo",
    #     assigner_model="gpt-4o",
    #     use_azure_openai=use_azure_openai,
    #     use_vllm=use_vllm,
    #     log_level=log_level
    # )

    # Use nuggetizer1 for this example
    nuggetizer = nuggetizer1

    # Extract and score nuggets
    print("\n📝 Extracting and scoring nuggets...")
    create_start = time.time()
    scored_nuggets = nuggetizer.create(request)
    create_time = time.time() - create_start
    print(f"Found {len(scored_nuggets)} nuggets (took {create_time:.2f}s):")
    for i, nugget in enumerate(scored_nuggets, 1):
        importance_emoji = "⭐" if nugget.importance == "vital" else "✔️"
        print(
            f"{i}. {importance_emoji} {nugget.text} (Importance: {nugget.importance})"
        )

    # Assign nuggets to documents
    print("\n🎯 Assigning nuggets to documents...")
    assign_start = time.time()
    for doc in request.documents:
        print(f"\nDocument: {doc.docid}")
        print("Segment:", doc.segment)
        assigned_nuggets = nuggetizer.assign(
            request.query.text, doc.segment, scored_nuggets
        )
        print("\nAssignments:")
        for nugget in assigned_nuggets:
            importance_emoji = "⭐" if nugget.importance == "vital" else "✨"
            assignment_emoji = {
                "support": "✅",
                "partial_support": "🟡",
                "not_support": "❌",
            }.get(nugget.assignment, "❓")
            print(f"{nugget.text}")
            print(f"  Importance: {nugget.importance} {importance_emoji}")
            print(f"  Assignment: {nugget.assignment} {assignment_emoji}")

        # Calculate metrics for this document
        nugget_list = [
            {"text": n.text, "importance": n.importance, "assignment": n.assignment}
            for n in assigned_nuggets
        ]
        metrics = calculate_nugget_scores(request.query.qid, nugget_list)
        print("\nMetrics:")
        print(f"  Strict Vital Score: {metrics.strict_vital_score:.2f}")
        print(f"  Strict All Score: {metrics.strict_all_score:.2f}")
        print(f"  Vital Score: {metrics.vital_score:.2f}")
        print(f"  All Score: {metrics.all_score:.2f}")

    assign_time = time.time() - assign_start
    total_time = time.time() - start_time
    print("\n⏱️ Timing Summary:")
    print(f"  Creation time: {create_time:.2f}s")
    print(f"  Assignment time: {assign_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")


def main():
    """Run the e2e example."""
    parser = argparse.ArgumentParser(description="Run the e2e example")
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--log_level", type=int, default=0, help="Log level")
    args = parser.parse_args()

    print("🔧 Starting E2E Nuggetizer Example...")
    print(f"Using model: {args.model}")
    request = create_sample_request()
    process_request(
        request, args.model, args.use_azure_openai, args.use_vllm, args.log_level
    )
    print("\n✨ Example completed!")


if __name__ == "__main__":
    main()
