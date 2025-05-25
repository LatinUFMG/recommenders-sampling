# Sampling Experiments

---

This project executes experiments for evaluating different sampling strategies in recommender systems, by running multiple simulated exploration biases using the [KuaiRec dataset](https://kuairec.com/).

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

Navigate to the root directory of the project (where `setup.py` is located) and install the project dependencies using pip:

```bash
pip install -e .
```

## Usage

---

This project runs sampling experiments with configurable parameters.

### Running the Experiment

To run the experiments, execute the main script with the desired arguments.

```bash
python run_kuai.py [OPTIONS]
```

### Arguments

* `-d`, `--dt`: **Execution date** in `YYYY-MM-DD HH:MM` format. Use this to continue an experiment from a previous execution. If not provided, a new execution will start with the current timestamp.
    * **Default**: `""` (empty string, which triggers a new execution)
* `--ignore_coverage_filter`: A **flag** to disable the filter that keeps only items with 100% user coverage in the test set. If this flag is present, the filter will **not** be applied.
    * **Default**: `True` (meaning the filter is applied by default)

#### Example: Starting a new experiment

First, navigate to script folder:

```bash
cd experiments/sampling
```

Then, to start a new experiment, simply run:

```bash
python run_kuai.py --ignore_coverage_filter
```

This will create a new results directory with the current timestamp.

#### Example: Continuing an experiment

To continue an experiment from a specific date and time (e.g., from `2025-04-30 23:59`), run:

```bash
python run_kuai.py --ignore_coverage_filter -d "2025-04-30 23:59"
```

## Plotting Results

---

The `paper_plots.ipynb` Jupyter Notebook is provided to visualize the results of our experiments.

### Running the Notebook

1.  **Activate your Conda environment**:
    ```bash
    conda activate sampling_experiments
    ```

2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3.  Open the `paper_plots.ipynb` notebook.

4.  **Update the `CURRENT_RESULT_PATH` variable**:

    Before running the cells, you **must** update the `CURRENT_RESULT_PATH` variable within the notebook to point to the actual path of your experiment results. This path will be similar to the example below, but the timestamp will match your experiment's execution date and time.

    Locate the line in first cell.

    ```python
    CURRENT_RESULT_PATH = "..."
    ```

    Change it to reflect the specific timestamp of the experiment run you wish to plot. For example, if your experiment generated results in a directory named `2025-04-30 23:59`, you would update the line to:

    ```python
    CURRENT_RESULT_PATH = "./data/KuaiRec/result/2025-04-30 23:59/2025-04-30 23:59_{}.parquet"
    ```

    Remember that the `DT_FORMAT` for the results path is `YYYY-MM-DD HH:MM`.

5.  Run all cells in the notebook to generate the plots.
