# Rag Evaluation Framework
A research prototype to evaluate retrieval-augmented generation (RAG) models, embeddings, and prompt strategies.


## Project structure

```graphql
rag-evaluation-framework/
├── data/                   # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/              # Jupyter notebooks
│   └── rag_evaluation.ipynb
├── src/                    # Python modules / scripts
│   └── rag_eval/           # Namespaced package
│       ├── __init__.py     # Package initializer
│       ├── embeddings.py   # Embedding functions
│       ├── retrieval.py    # FAISS / retrieval utilities
│       ├── evaluation.py   # Metrics and evaluation scripts
│       └── utils.py        # Helper functions
├── results/                # Outputs, logs, metrics, plots
├── requirements.txt        # Top-level dependencies (locked versions)
├── pyproject.toml          # Build / package configuration
├── README.md               # Project landing page
└── .gitignore
```

## Project Setup

This project uses a `src/` layout with editable install for development. Follow these steps to set up your environment and run the notebooks.

### 1. Clone the Repository

```bash
git clone https://github.com/jsandino/rag-evaluation-framework.git
cd rag-evaluation-framework
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Package in Editable Mode

```bash
pip install -e .
```
This allows you to import the package directly from `src/rag_eval` and ensures changes in source files are immediately reflected.

### 5. Open Notebooks

- Open the Jupyter notebook(s) in VS Code.
- Make sure the kernel is using the .venv Python interpreter.
- Run the notebook cells as normal.

This setup ensures reproducibility and a clean development workflow.