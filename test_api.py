# -*- coding: utf-8 -*-
"""
Example usage of the RAG PDF API
"""

import requests
import json


def test_api():
    base_url = "http://localhost:8000"
    
    # 1. Check status
    print("🔍 Checking API status...")
    response = requests.get(f"{base_url}/status")
    print(f"Status: {response.json()}")
    
    # 2. Upload PDF (example - replace with actual PDF file)
    print("\n📤 Uploading PDF...")
    # with open("path/to/your/file.pdf", "rb") as f:
    #     files = {"file": ("document.pdf", f, "application/pdf")}
    #     response = requests.post(f"{base_url}/upload", files=files)
    #     print(f"Upload result: {response.json()}")
    
    # 3. Query without context
    print("\n❓ Querying without context...")
    query_data = {"question": "What is the main idea of the paper?"}
    response = requests.post(f"{base_url}/query", json=query_data)
    print(f"Answer: {response.json()}")
    
    # 4. Query with context
    print("\n❓ Querying with context...")
    query_data = {"question": "What are the key findings?", "include_context": True}
    response = requests.post(f"{base_url}/query", json=query_data)
    result = response.json()
    print(f"Answer: {result['answer']}")
    print(f"Context: {json.dumps(result['context'], indent=2)}")


if __name__ == "__main__":
    test_api()

