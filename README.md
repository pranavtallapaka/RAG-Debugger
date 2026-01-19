# RAG Quality Debugger

A tool for debugging and evaluating the quality of Retrieval-Augmented Generation (RAG) systems.

## Overview

This project provides a comprehensive debugging toolkit for RAG systems, focusing on three key quality dimensions:

- **Relevance**: Evaluates how relevant retrieved documents are to queries
- **Redundancy**: Detects redundant or duplicate information across retrieved documents
- **Coverage**: Assesses how well retrieved content covers query requirements

## Project Structure

```
.
├── evaluators/           # Evaluation modules
│   ├── relevance/        # Relevance evaluation
│   ├── redundancy/       # Redundancy detection
│   └── coverage/         # Coverage assessment
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

### Testing with Example Data

Multiple example datasets are provided for testing different scenarios:

- **`example_data_all.txt`**: All examples in human-readable format
- **`example_data_parsed.py`**: Python variables with all examples
- **`example_data.txt`**: Original example (for backward compatibility)

**Available Examples:**

1. **Example 1: Mixed Quality (ML)** - Mixed quality retrieval with some relevant, some redundant, and some irrelevant chunks. Good for testing all three evaluators.
   - Query: About supervised vs unsupervised learning
   - Chunks: 2 relevant, 2 redundant, 1 irrelevant

2. **Example 2: High Redundancy (Climate)** - Multiple chunks with very similar content. Should trigger high redundancy warnings.
   - Query: About causes of climate change
   - Chunks: All relevant but highly redundant

3. **Example 3: Low Relevance (Quantum)** - Query about quantum computing, but chunks are mostly about classical computing. Should trigger missing context warnings.
   - Query: About quantum computers and qubits
   - Chunks: Mostly irrelevant (classical computing, general physics)

4. **Example 4: Good Quality (Neural Networks)** - All chunks are highly relevant with good coverage. Low redundancy with complementary information.
   - Query: About neural network components
   - Chunks: All relevant, good coverage, low redundancy

**Using Examples in the App:**

Select an example from the dropdown menu in the Streamlit app, or manually copy query and chunks from `example_data_all.txt`.

## Development

This is a skeleton project. Implementation details for each evaluator are pending.

## License

[Add your license here]

