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
├── README.md                # Usage instructions
├── main.py                  # Main program entry point
├── requirements.txt         # Environment dependencies
└── supercell_construct.py   # Generate supercell grouping labels
```

> Note: Some excessively large data files are not included in `data` directory in this repository.
> The complete dataset can be downloaded from Zenodo:  
> **[Code for scYOU](https://zenodo.org/records/18756874)**

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
    "num_protos": [5],           # Number of clusters (true cluster count)

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

## Note
* If you are using a new dataset, please generate the GO similarity matrix and the supercell grouping labels before running the code (refer to the "Materials and Methods" section in the article). We have provided the file **`supercell_construct.py`** here for generating the supercell grouping labels.
