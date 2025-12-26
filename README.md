# **`README.md`**

# Who Connects Global Aid? The Hidden Geometry of 10 Million Transactions

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.17243-b31b1b.svg)](https://arxiv.org/abs/2512.17243)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2512.17243)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Discipline](https://img.shields.io/badge/Discipline-Network%20Science%20%7C%20Development%20Economics-00529B)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Data Sources](https://img.shields.io/badge/Data-IATI%20Registry-lightgrey)](https://iatiregistry.org/)
[![Data Sources](https://img.shields.io/badge/Data-Web%20Hyperlink%20Graph-lightgrey)](https://commoncrawl.org/)
[![Core Method](https://img.shields.io/badge/Method-Bipartite%20Projection-orange)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Analysis](https://img.shields.io/badge/Analysis-Node2Vec%20Embedding-red)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Validation](https://img.shields.io/badge/Validation-PageRank%20Correlation-green)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Robustness](https://img.shields.io/badge/Robustness-Sensitivity%20Analysis-yellow)](https://github.com/chirindaopensource/who_connects_global_aid)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/scipy-%230054A6.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![NetworkX](https://img.shields.io/badge/networkx-%230054A6.svg?style=flat&logo=networkx&logoColor=white)](https://networkx.org/)
[![UMAP](https://img.shields.io/badge/umap-%230054A6.svg?style=flat&logo=python&logoColor=white)](https://umap-learn.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/who_connects_global_aid`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Who Connects Global Aid? The Hidden Geometry of 10 Million Transactions"** by:

*   **Paul X. McCarthy** (League of Scholars, Sydney; UNSW)
*   **Xian Gong** (University of Technology Sydney)
*   **Marian-Andrei Rizoiu** (University of Technology Sydney)
*   **Paolo Boldi** (Università degli Studi di Milano)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and cleansing of massive IATI transaction logs to the rigorous construction of bipartite networks, high-dimensional structural embedding via Node2Vec, and the identification of critical "knowledge brokers" through centrality analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_global_aid_pipeline`](#key-callable-run_global_aid_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in McCarthy et al. (2025). The core of this repository is the iPython Notebook `who_connects_global_aid_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed to map the topological structure of the global aid ecosystem, moving beyond aggregate volume flows to reveal the hidden geometry of influence.

The paper addresses the structural complexity of the modern aid system, characterized by a "triple revolution" of new goals, instruments, and actors. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Cleanse and normalize over 10 million transaction records from the International Aid Transparency Initiative (IATI).
-   Construct a bipartite Provider-Receiver network and project it into a Provider-Provider co-investment graph.
-   Learn high-dimensional structural embeddings using Node2Vec to capture functional roles.
-   Reveal functional clusters (Humanitarian vs. Development) via UMAP dimensionality reduction.
-   Identify the "Solar System" of central actors using Hub Scores (HITS) and Betweenness Centrality.
-   Validate findings externally by correlating offline structural influence with online web authority (PageRank).

## Theoretical Background

The implemented methods combine techniques from Network Science, Graph Theory, and Machine Learning.

**1. Bipartite Network Construction:**
The system is modeled as a bipartite graph $G = (U, V, E)$, where $U$ represents Provider organisations and $V$ represents Receiver organisations. An edge $e_{uv}$ exists if a financial transaction occurs, weighted by frequency.

**2. One-Mode Projection:**
To analyze donor relationships, the bipartite graph is projected into a Provider-Provider co-occurrence network $P = MM^T$, where $M$ is the incidence matrix of providers appearing in shared contexts (countries or sectors).

**3. Structural Embedding (Node2Vec):**
The topology is encoded into a low-dimensional vector space $f: V \to \mathbb{R}^d$ by optimizing a Skip-gram objective over biased random walks:
$$ \max_f \sum_{u \in V} \log \Pr(N_S(u) | f(u)) $$
This captures structural equivalence, grouping actors with similar network roles regardless of direct connectivity.

**4. Centrality and Brokerage:**
Influence is quantified using HITS Hub Scores (for the "Solar System" ranking) and Betweenness Centrality (for identifying brokers):
$$ C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}} $$
This metric highlights actors like J-PAL and the Hewlett Foundation that bridge structural holes between disparate clusters.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/who_connects_global_aid/blob/main/who_connects_global_aid_summary_two.png" alt="Global Aid Network Analysis Process Summary" width="100%">
</div>

## Features

The provided iPython Notebook (`who_connects_global_aid_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 29 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (ER thresholds, Node2Vec hyperparameters, UMAP settings) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, type coercion feasibility, and referential integrity.
-   **Deterministic Entity Resolution:** Implements a robust TF-IDF blocking and Cosine Similarity pipeline to resolve organisation names to canonical identities.
-   **High-Performance Computing:** Utilizes Numba-accelerated random walk generation and sparse matrix algebra for scalability.
-   **Reproducible Artifacts:** Generates structured dictionaries, serializable outputs, and cryptographic manifests for every intermediate result.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-8):** Ingests raw IATI data, validates schemas, enforces temporal scope (1967-2025), and normalizes multi-valued fields.
2.  **Integration & ER (Tasks 9-10):** Joins transactions to activity contexts and performs entity resolution to construct a canonical organisation map.
3.  **Descriptive Analysis (Tasks 11-12):** Computes geographic transaction density and longitudinal instrument evolution.
4.  **Network Construction (Tasks 13-15):** Builds the bipartite graph and projects it into a provider co-occurrence network.
5.  **Topology & Embeddings (Tasks 16-17):** Generates Node2Vec embeddings and projects them to 2D using UMAP to reveal functional clusters.
6.  **Centrality & Ranking (Tasks 18-20):** Computes HITS and Betweenness centrality to construct the "Solar System" ranking.
7.  **Analysis & Validation (Tasks 21-26):** Analyzes subgroups (Universities/Foundations), characterizes broker networks (Hewlett), and correlates findings with web PageRank.
8.  **Orchestration & Provenance (Tasks 27-29):** Manages the end-to-end execution and packages reproducible outputs.

## Core Components (Notebook Structure)

The `who_connects_global_aid_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 29 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_global_aid_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_global_aid_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between cleansing, graph construction, and analysis modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `networkx`, `gensim`, `scikit-learn`, `umap-learn`, `numba`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/who_connects_global_aid.git
    cd who_connects_global_aid
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy networkx gensim scikit-learn umap-learn numba pyyaml
    ```

## Input Data Structure

The pipeline requires four primary DataFrames:
1.  **`df_transactions_raw`**: IATI transaction elements (Value, Date, Provider, Receiver).
2.  **`df_activities_raw`**: Activity metadata (Sectors, Countries, Instruments).
3.  **`df_organisations_raw`**: Master organisation list for entity resolution.
4.  **`df_web_links_raw`**: Web crawl data for external validation.

## Usage

The `who_connects_global_aid_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `execute_master_workflow` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    data_inputs = generate_synthetic_data()
    
    # 3. Execute the entire replication study.
    results = execute_master_workflow(
        df_transactions_raw=data_inputs["df_transactions_raw"],
        df_activities_raw=data_inputs["df_activities_raw"],
        df_organisations_raw=data_inputs["df_organisations_raw"],
        df_web_links_raw=data_inputs["df_web_links_raw"],
        config=config,
        output_root="./global_aid_study_output"
    )
    
    # 4. Access results
    if results.success:
        print(f"Validation Correlation: {results.baseline_results.artifacts['correlation'].pearson_r}")
```

## Output Structure

The pipeline returns a `MasterWorkflowResults` object containing:
-   **`baseline_results`**: A `PipelineResults` object with all artifacts from the primary run (Graph, Embeddings, Centrality Tables).
-   **`robustness_results`**: A `RobustnessArtifact` containing sensitivity analysis metrics.
-   **`provenance_artifact`**: A `ProvenanceArtifact` with cryptographic hashes and metadata summaries.

## Project Structure

```
who_connects_global_aid/
│
├── who_connects_global_aid_draft.ipynb  # Main implementation notebook
├── config.yaml                          # Master configuration file
├── requirements.txt                     # Python package dependencies
│
├── LICENSE                              # MIT Project License File
└── README.md                            # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Scope:** `start_year`, `end_year`.
-   **Entity Resolution:** `fuzzy_matching` threshold.
-   **Node2Vec:** `p`, `q`, `walk_length`, `num_walks`.
-   **UMAP:** `n_neighbors`, `min_dist`.
-   **Projection:** Context definitions (Country vs. Sector).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Temporal Dynamics:** Analyzing the evolution of the network topology over sliding windows.
-   **Multiplex Networks:** Modeling different financial instruments (Grants vs. Loans) as distinct layers.
-   **Impact Analysis:** Correlating network centrality with aid effectiveness outcomes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{mccarthy2025who,
  title={Who Connects Global Aid? The Hidden Geometry of 10 Million Transactions},
  author={McCarthy, Paul X. and Gong, Xian and Rizoiu, Marian-Andrei and Boldi, Paolo},
  journal={arXiv preprint arXiv:2512.17243},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). Global Aid Network Analysis Pipeline: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/who_connects_global_aid
```

## Acknowledgments

-   Credit to **Paul X. McCarthy, Xian Gong, Marian-Andrei Rizoiu, and Paolo Boldi** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, NetworkX, Gensim, and UMAP**.

--

*This README was generated based on the structure and content of the `who_connects_global_aid_draft.ipynb` notebook and follows best practices for research software documentation.*
