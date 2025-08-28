#!/usr/bin/env python3
"""
Simple test script to verify vLLM support in nuggetizer.
"""

import os
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer


def test_vllm_support():
    """Test vLLM support with a simple example."""

    # Create a simple test request
    query = Query(qid="test-1", text="What is Python?")
    documents = [
        Document(
            docid="doc1",
            segment="Python is a high-level programming language known for its simplicity and readability.",
        ),
        Document(
            docid="doc2",
            segment="Python was created by Guido van Rossum and released in 1991.",
        ),
    ]
    request = Request(query=query, documents=documents)

    print("🧪 Testing vLLM support...")
    print(f"VLLM_API_BASE: {os.getenv('VLLM_API_BASE', 'Not set')}")

    # Set VLLM API base URL
    os.environ["VLLM_API_BASE"] = "http://localhost:8001/v1"
    print(f"Set VLLM_API_BASE to: {os.environ['VLLM_API_BASE']}")

    try:
        # Test with vLLM
        nuggetizer = Nuggetizer(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",  # Replace with your actual model name
            use_vllm=True,
            log_level=1,
        )

        print("✅ Nuggetizer initialized with vLLM support")

        # Test nugget creation
        print("📝 Testing nugget creation...")
        scored_nuggets = nuggetizer.create(request)

        print(f"✅ Successfully created {len(scored_nuggets)} nuggets:")
        for i, nugget in enumerate(scored_nuggets, 1):
            print(f"  {i}. {nugget.text} (Importance: {nugget.importance})")

        # Test nugget assignment
        print("\n🎯 Testing nugget assignment...")
        assigned_nuggets = nuggetizer.assign(
            query.text, documents[0].segment, scored_nuggets
        )

        print("✅ Successfully assigned nuggets:")
        for nugget in assigned_nuggets:
            print(f"  - {nugget.text}")
            print(f"    Importance: {nugget.importance}")
            print(f"    Assignment: {nugget.assignment}")

        print("\n🎉 vLLM support test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing vLLM support: {str(e)}")
        print("\n💡 Make sure to:")
        print("  1. Set VLLM_API_BASE in your .env file")
        print("  2. Replace 'your-model-name' with your actual model name")
        print("  3. Ensure your vLLM server is running and accessible")


if __name__ == "__main__":
    test_vllm_support()
