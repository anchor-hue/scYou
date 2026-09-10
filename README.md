# scYOU

## 1. Project Structure

```markdown
scYOU/
├── config/
│   └── config.py            # Centralized configuration file
├── data/
│   ├── expression/          # Protein expression matrices
│   ├── GO/                  # Gene Ontology similarity constraints
│   ├── meta/                # Metadata files containing ground-truth labels
│   └── supercell/           # Supercell constraints
├── src/
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Model architecture definitions
│   ├── trainer.py           # Training pipeline
│   └── utils.py             # Utility functions
├── README.md                         # Usage instructions
├── main.py                           # Main program entry point
├── requirements.txt                  # Environment dependencies
├── supercell_construct.py            # Generate supercell grouping labels
└── estimate_k_eigengap.py      # Label-free cluster-number estimation by spectral eigengap
```

> Note: Some excessively large data files are not included in the `data` directory of this repository.
> The complete processed datasets are archived at Zenodo:  
> **[https://zenodo.org/records/18756874](https://zenodo.org/records/18756874)**

---

## 2. Quick Start

### 2.1 Install Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

---

### 2.2 Configure Parameters and File Paths

All parameters and file paths should be configured in **`config.py`**.

---

#### Model Configuration

```python
# ===================== Model Configuration =====================
MODEL_CONFIG = {
    "cell_embed_dim": 32,          # Dimension of cell embeddings
    "protein_embed_dim": 32,       # Dimension of protein embeddings
    "weight_decay": 1e-5,          # Weight decay for optimization
    "max_pretrain_epochs": 2000,   # Maximum number of pretraining epochs
    "max_train_epochs": 2000,      # Maximum number of training epochs
    "convergence_patience": 15,    # Early stopping patience
    "convergence_threshold": 1e-4, # Convergence threshold
    "tau": 0.5,                    # Temperature parameter for contrastive loss
    "alpha_cluster": 1.0           # Alpha parameter for clustering layer
}
```

---

#### Dataset Configuration

*(Example: Montalvo dataset)*

```python
# ===================== File Path Configuration =====================
FILE_PATHS = {
    # Expression matrix
    "expression_matrix": "./data/expression/expression_Montalvo.csv",

    # GO similarity matrix
    "go_similarity": "./data/GO/GO_Montalvo.csv",

    # Cell label metadata
    "cell_labels": "./data/meta/meta_Montalvo.csv",

    # Supercell label file
    "supercell_labels": "./data/supercell/supercell_Montalvo.csv",

    # Output directories
    "results_base_dir": "./grid_search_results/",
    "loss_plots_dir": "./loss_curves/",
    "embeddings_dir": "./embeddings/"
}
```

---

#### Label Column Configuration

```python
# ===================== Label Column Configuration =====================
LABEL_COLUMNS = {
    # Adjust according to the dataset used
    "cell_type_column": "Cell_type",     # For Montalvo dataset
    "supercell_column": "supercell_label"
}
```

---

#### Other Hyperparameters

```python
# ===================== Grid Search Parameters =====================
GRID_SEARCH_PARAMS = {

    # Montalvo dataset settings
    "n_top_var": [501],          # Number of highly variable proteins
    "num_protos": [5],           # Number of clusters for the known-K setting

    "alpha": [1.0],
    "beta": [0.1],
    "learning_rate": [0.001],
    "tol": [0.005],

    # General settings
    "gamma": [1.0],
    "delta": [1.0],
    "update_interval": [10],
    "seed": [9842]
}
```

---

#### Environment Configuration

```python
# ===================== Environment Configuration =====================
ENV_CONFIG = {
    "global_seed": 42,
    "device": torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
    "result_output_dir": "./grid_search_results/"
}
```

---

### 2.3 Run the Main Program

Execute the main script:

```bash
python main.py
```

---

### 2.4 Check Results

After execution:

* **clustering metrics** will be printed in the console.
* **Loss curves** and **Learned embeddings** will be saved in `./grid_search_results/`.

---

## 3. Label-free Cluster-number Estimation

For analyses in which the number of cell populations is not assumed to be known in advance, scYOU provides a spectral eigengap procedure for estimating the cluster number $K$ directly from the processed protein-expression matrix.

The implementation is provided in **`estimate_k_eigengap.py`**. The procedure:

1. reads each protein-by-cell expression matrix and converts it to a cell-by-protein matrix;
2. replaces missing values with zero and removes zero-variance proteins;
3. projects cells into a PCA space using up to 32 principal components;
4. constructs a symmetric k-nearest-neighbor graph with an adaptive neighborhood size;
5. computes the normalized graph Laplacian; and
6. estimates $K$ as the value that maximizes the spectral eigengap $\lambda_{K+1}-\lambda_K$ over the specified search range.

This procedure does not use cell-type annotations, ARI, NMI, or other ground-truth clustering information.

### 3.1 Example command

```bash
python estimate_k_eigengap.py \
  --input-dir ./data/expression \
  --output-dir ./cluster_number_results \
  --pattern "*.csv" \
  --k-min 2 \
  --k-max 10 \
  --n-components 32 \
  --seed 42
```

The default settings correspond to the label-free cluster-number estimation described in the manuscript. For a dataset with $n$ cells, the neighborhood size is selected adaptively as

```text
min(30, max(15, floor(sqrt(n))))
```

subject to the number of available cells.

The estimated cluster numbers are written to:

```text
./cluster_number_results/cluster_number_summary.csv
```

The output table contains the dataset name, numbers of cells and proteins, PCA dimension, explained-variance fraction, KNN neighborhood size, and the estimated cluster number.

### 3.2 Using the estimated cluster number in scYOU

The `num_protos` value in `config.py` controls the number of clusters used by the clustering module.

- **Known-K benchmark:** set `num_protos` to the annotated number of cell populations.
- **Label-free analysis:** first run `estimate_k_eigengap.py`, then set `num_protos` to the corresponding `estimated_K` reported in `cluster_number_summary.csv`.


## Note
* If you are using a new dataset, please generate the GO similarity matrix and the supercell grouping labels (refer to the "Materials and Methods" section in the article) before running the code. We have provided the file **`supercell_construct.py`** here for generating the supercell grouping labels.
