"""Debug runner for Lab 1 tests.
This script imports the test function and runs it inside a try/except so
any exception and full traceback is printed to the action logs.
"""
import traceback
import sys

# Ensure repository root is on sys.path
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    # Import test module and run the test function
    from test.lab_1 import test_tokenizers

    print("Running test_tokenizers()...\n")
    test_tokenizers()
    print("\nTESTS COMPLETED: test_tokenizers() finished without exception.")

    # Run the lab 2 portion by invoking main section (re-import module to execute top-level code)
    print("\nRunning lab 2 section (vectorizer demo)...\n")
    # Execute the file as a script to run the top-level Lab 2 demo
    with open(os.path.join(root, 'test', 'lab_1.py'), 'r', encoding='utf-8') as f:
        code = f.read()
    exec(compile(code, os.path.join(root, 'test', 'lab_1.py'), 'exec'), {})

except Exception:
    print("\nEXCEPTION DURING TEST RUN:\n")
    traceback.print_exc()
    # Exit non-zero so the workflow shows failure, but logs will include trace
    sys.exit(1)
