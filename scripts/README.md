# Scripts

This directory contains executable scripts for running the complaint analysis pipeline.

## Available Scripts

### `run_task1.py`
Runs Task 1: Data preprocessing and cleaning.
```bash
python scripts/run_task1.py
```

### `run_task2.py`
Runs Task 2: Chunking, embedding, and vector store creation.
```bash
python scripts/run_task2.py
```

Make sure to run Task 1 before Task 2, as Task 2 depends on the output of Task 1.

