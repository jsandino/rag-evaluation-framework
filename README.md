# rag-evaluation-framework
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
│   ├── embeddings.py       # Embedding functions
│   ├── retrieval.py        # FAISS / retrieval utilities
│   ├── evaluation.py       # Metrics and evaluation scripts
│   └── utils.py            # Helper functions
├── results/                # Outputs, logs, metrics, plots
├── requirements.txt        # Top-level dependencies (locked versions)
├── README.md               # Project landing page
└── .gitignore
```