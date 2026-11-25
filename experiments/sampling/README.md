# Sampling Experiments

---

This project executes experiments for evaluating different sampling strategies in recommender systems. It supports running multiple simulated exploration biases using the [KuaiRec dataset](https://kuairec.com/), as well as evaluating datasets without simulated biases—such as [COAT](https://www.cs.cornell.edu/~schnabts/mnar/) and [R3](http://infolab.stanford.edu/~ullman/mining/2009/YahooData.pdf)—via dataset modules.

## Installation

---

Follow these steps to set up your environment and install the necessary dependencies.

### 1. Create Conda Environment

First, create a new Conda environment with Python 3.10:

```bash
conda create -n sampling_experiments python=3.10
```

### 2. Activate Environment

Activate the newly created Conda environment:

```bash
conda activate sampling_experiments
```

### 3. Install Dependencies

First, downgrade pip to version 25.0, as this is required to ensure compatibility with LightFM:

```bash
pip install --force pip==25
```

Then, navigate to the root directory of the project (where `setup.py` is located) and install the project dependencies using pip:

```bash
pip install -e .[gpu]
```

## Usage

---

This project runs sampling experiments with configurable parameters. Before running any script, navigate to the experiment folder:

```bash
cd experiments/sampling
```

### 1. Simulated Exploration Bias Experiments (`run_kuai.py`)

Use `run_kuai.py` to execute experiments with multiple simulated exploration biases on the KuaiRec dataset:

```bash
python run_kuai.py [OPTIONS]
```

#### Arguments

- `-d`, `--dt`: **Execution date** in `YYYY-MM-DD HH:MM` format. Use this to continue an experiment from a previous execution. If not provided, a new execution will start with the current timestamp.
  - **Default**: `""` (empty string, which triggers a new execution)
- `--ignore_coverage_filter`: A **flag** to disable the filter that keeps only items with 100% user coverage in the test set. If this flag is present, the filter will **not** be applied.
  - **Default**: `True` (meaning the filter is applied by default)

#### Examples

**Starting a new experiment:**

```bash
python run_kuai.py --ignore_coverage_filter
```

This will create a new results directory with the current timestamp.

**Continuing an experiment:**

To continue an experiment from a specific date and time (e.g., from `2025-04-30 23:59`), run:

```bash
python run_kuai.py --ignore_coverage_filter -d "2025-04-30 23:59"
```

---

### 2. Non-Simulated Dataset Experiments (`run_generic.py`)

Use `run_generic.py` to evaluate sampling strategies on datasets (such as COAT and R3) without generating exploration bias simulations:

```bash
python run_generic.py --dataset <DATASET_NAME> [OPTIONS]
```

#### Arguments

- `--dataset` **(Required)**: Name of the dataset module (e.g., `'coat'`, `'r3'`, `'kuai'`, `'criteo'`) available in `recommenders.datasets`.
- `-d`, `--dt`: **Execution date** in `YYYY-MM-DD HH:MM` format. Use this to continue an experiment from a previous execution. If not provided, a new execution will start with the current timestamp.
  - **Default**: `""` (empty string, which triggers a new execution)
- `--ignore_coverage_filter`: A **flag** to disable the filter that keeps only items with 100% user coverage. If this flag is present, the filter will **not** be applied.
  - **Default**: `True` (meaning the filter is applied by default)
- `--hparam_seed`: An integer seed. If set, disables tuning and samples hyperparameters once using this seed. Requires a first execution with tuning enabled.
  - **Default**: `None`

#### Examples

**Starting a new experiment (e.g., COAT):**

```bash
python run_generic.py --dataset coat --ignore_coverage_filter
```

**Running with a fixed hyperparameter seed (e.g., R3):**

```bash
python run_generic.py --dataset r3 --ignore_coverage_filter --hparam_seed 42
```

**Continuing an experiment:**

```bash
python run_generic.py --dataset coat --ignore_coverage_filter -d "2025-04-30 23:59"
```

## Plotting Results

---

The following Jupyter Notebooks are provided to visualize the results of our experiments:

- `recsys25_plots.ipynb`: Generates the result plots for the RecSys '25 paper.
- `tors26_kuai_plots.ipynb`: Generates plots for the KuaiRec experiments for TORS '26.
- `tors26_model_comparison_by_sampler.ipynb`: Generates model comparison plots grouped by sampler for TORS '26.
- `tors26_extra_plots.ipynb`: Generates plots for the COAT/R3 experiments for TORS '26.

### Running the Notebooks

1. **Activate your Conda environment**:
   ```bash
   conda activate sampling_experiments
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. Open the desired plotting notebook (e.g., `recsys25_plots.ipynb` or one of the `tors26_*.ipynb` notebooks).

4. **Update the `CURRENT_RESULT_PATH` variable**:

   Before running the cells, you **must** update the `CURRENT_RESULT_PATH` variable within the notebook to point to the actual path of your experiment results. This path will be similar to the example below, matching your dataset directory and the experiment's execution date and time.

   Locate the line in the first cell:

   ```python
   CURRENT_RESULT_PATH = "..."
   ```

   Change it to reflect the specific dataset and timestamp of the experiment run you wish to plot. For example, if your experiment generated results in `./data/KuaiRec/result/2025-04-30 23:59`, you would update the line to:

   ```python
   CURRENT_RESULT_PATH = "./data/KuaiRec/result/2025-04-30 23:59/2025-04-30 23:59_{}.parquet"
   ```

   Remember that the `DT_FORMAT` for the results path is `YYYY-MM-DD HH:MM`.

5. Run all cells in the notebook to generate the plots.
