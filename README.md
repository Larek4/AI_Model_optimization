# Optimizing AI Model Training According to Hardware Constraints

## Table of Contents

1.  [Introduction](https://www.google.com/search?q=%23introduction)
2.  [Project Goals](https://www.google.com/search?q=%23project-goals)
3.  [Methods and Technologies Used](https://www.google.com/search?q=%23methods-and-technologies-used)
      * [Deep Learning Framework](https://www.google.com/search?q=%23deep-learning-framework)
      * [Optimization Framework](https://www.google.com/search?q=%23optimization-framework)
      * [Optimization Techniques Implemented](https://www.google.com/search?q=%23optimization-techniques-implemented)
      * [Performance Monitoring](https://www.google.com/search?q=%23performance-monitoring)
      * [Data Handling](https://www.google.com/search?q=%23data-handling)
4.  [Project Structure](https://www.google.com/search?q=%23project-structure)
5.  [Installation](https://www.google.com/search?q=%23installation)
6.  [Usage](https://www.google.com/search?q=%23usage)
7.  [Results Summary](https://www.google.com/search?q=%23results-summary)
8.  [Key Learnings and Challenges](https://www.google.com/search?q=%23key-learnings-and-challenges)
9.  [Credits](https://www.google.com/search?q=%23credits)
10. [License](https://www.google.com/search?q=%23license)

## 1\. Introduction

Training large AI models presents significant challenges due to hardware constraints, often leading to prolonged training times, out-of-memory errors, and inefficient hardware utilization. This project addresses these issues by focusing on optimizing AI model training for faster experimentation, reduced energy consumption, and maximum utilization of existing computational infrastructure, particularly on limited resources.

We used the Intel Image Classification dataset, which consists of RGB JPEG images across six categories, initially standardized to $150 \\times 150$ pixels. For the deep learning model, SqueezeNet version 1.1 was selected due to its balance of accuracy (0.89) and compact size (4.8 MB), which aligned well with the project's optimization goals under hardware limitations. The primary development and experimentation environment was Google Colab's T4 GPU, which offered crucial GPU access despite daily usage limits.

## 2\. Project Goals

The main objectives of this project were to:

  * Achieve stable training and improved accuracy for the SqueezeNet 1.1 model.
  * Maximize CPU utilization concurrently, aiming for balanced hardware usage, as initial observations via TensorBoard showed significant CPU underutilization (around 20%).
  * Develop real-world, hardware-aware AI engineering skills.
  * Implement and integrate hyperparameter optimization and resolution tuning into a combined pipeline.

## 3\. Methods and Technologies Used

To address training instability, suboptimal accuracy, and inefficient hardware utilization, we employed a suite of software optimization techniques.

### Deep Learning Framework

  * **PyTorch**: The project was implemented in Python within the Google Colab environment, leveraging the PyTorch framework.

### Optimization Framework

  * **Optuna**: Chosen as an open-source hyperparameter optimization framework for its efficient, automated search capabilities. Optuna explores a search space of hyperparameters through "trials," defining an "objective function" to maximize (e.g., validation accuracy). This approach proved superior to manual tuning, grid search, or random search due to its adaptiveness and efficiency, which was crucial given Google Colab's GPU limitations.
      * **Tree-structured Parzen Estimator (TPE) Sampler**: Both optimization studies utilized Optuna's TPE sampler. TPE is a Bayesian optimization algorithm that builds probability distributions based on past evaluated trials to suggest new hyperparameters more likely to improve the objective. This directed search leads to faster convergence to optimal solutions, directly aligning with the goal of faster experimentation and maximizing the utility of existing computational infrastructure.

### Optimization Techniques Implemented

  * **Hyperparameter Optimization** (`optuna_optimizer.py`): Using the `OptunaTuner` class, validation accuracy was maximized by tuning the learning rate, batch size, and epochs. The batch size search space was expanded to include larger values.
  * **Resolution Tuning** (`resolution_tuner.py`): The `ResolutionTuner` class dynamically optimized image resolution (e.g., [144, 160, 176, 192]), also aiming to maximize validation accuracy.
  * **Automatic Mixed Precision (AMP)**: Implemented within `trainer_2.py` using `torch.amp.autocast` and `torch.cuda.amp.GradScaler`. AMP improves training speed and reduces memory footprint by using lower precision operations and managing gradient scaling. It was a key enabler for achieving high GPU saturation.
  * **Learning Rate Scheduler (CosineAnnealingLR)**: Employed to adjust the learning rate during training.
  * **DataLoader with `num_workers` and `pin_memory=True`**: Used for efficient data loading, contributing to maximum CPU utilization. `pin_memory=True` was added for DataLoaders.
  * **Adam Optimizer**: Used as the primary optimizer for training the model.
  * **SqueezeNet1\_1\_Weights.DEFAULT**: Utilized for initializing the SqueezeNet model with pre-trained weights.

### Performance Monitoring

  * **TorchProfiler** (`profiler.py`): The `TorchProfiler` class captured detailed CPU and CUDA activity using `torch.profiler` and monitored system resources via `psutil`. This provided crucial insights into hardware utilization.
  * **TensorBoard**: Used for profiling, as mentioned in the key learnings, and for visualizing training results.

### Data Handling

  * **`dataset.py`**: Handled dataset downloading via `kagglehub` and applied image transformations, including dynamic resizing.
  * **`utility.py`**: Provided helper functions for inference and evaluation.

## 4\. Project Structure

The project is organized into a modular structure to ensure maintainability and clarity.

```
your_project_folder/
├── modules/
│   ├── dataset.py            # Handles Intel Image Classification dataset loading and transformations.
│   ├── trainer_2.py          # Custom Trainer class with AMP, LR scheduler, etc.
│   ├── utility.py            # Helper functions for inference, evaluation, plotting.
│   └── profiler.py           # TorchProfiler for performance monitoring.
├── optuna_optimizer.py       # OptunaTuner class for hyperparameter optimization.
├── resolution_tuner.py       # ResolutionTuner class for image resolution optimization.
├── requirements.txt          # Lists all necessary Python packages.
└── main_pipeline.ipynb       # Main Jupyter Notebook to run scenarios.
```

**Key Points:**

  * The `modules` folder contains core utility classes.
  * `optuna_optimizer.py`, `resolution_tuner.py`, and `requirements.txt` must be in the **same root directory as your `modules` folder** and your main execution script (e.g., `main_pipeline.ipynb`), but *not inside* the `modules` folder itself.
  * The folder named `others` (not explicitly shown in structure, but mentioned in source) contains the saved model and profiler traces.

## 5\. Installation

To set up the environment and run the project, it is recommended to use a virtual environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your_project_folder.git
    cd your_project_folder
    ```

2.  **Install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    pip install accelerate==0.21.0 optuna tensorboard
    ```

    The `requirements.txt` file lists:

      * `ipython`
      * `ipywidgets`
      * `kagglehub`
      * `plotly`
      * `torch`
      * `torchmetrics`
      * `torchvision`

## 6\. Usage

The `main_pipeline.ipynb` notebook orchestrates the different optimization scenarios.

1.  **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook "main pipeline.ipynb"
    ```

    or open it in Google Colab directly.

2.  **Configure Scenarios:**
    Inside `main_pipeline.ipynb`, you can control which scenarios run by setting the following boolean flags in the `CONFIGURATION` section:

    ```python
    # ==== CONFIGURATION ====
    RUN_BASELINE = False
    RUN_RESOLUTION_TUNING = False
    RUN_OPTUNA_HP_TUNING = True
    PERFORM_DETAILED_PROFILING = True # Set this to True only if you need a detailed profiler trace for a specific scenario
    ```

      * Set `RUN_BASELINE = True` to run the unoptimized baseline scenario.
      * Set `RUN_RESOLUTION_TUNING = True` to run the resolution optimization.
      * Set `RUN_OPTUNA_HP_TUNING = True` to run the hyperparameter optimization.
      * `PERFORM_DETAILED_PROFILING` can be toggled for in-depth profiling.

3.  **Run the Notebook:**
    Execute all cells in the `main_pipeline.ipynb` notebook. The script will perform the configured training scenarios, plot the results (saving `training_loss_comparison.png` and `training_accuracy_comparison.png`), and save the final trained model (`intel_model.pt`).

## 7\. Results Summary

The project compared the performance of SqueezeNet 1.1 across three distinct scenarios:

| Metric / Scenario       | Baseline (10 Ep) | Res. Opt. (10 Ep) | HP Opt. (20 Ep) |
| :---------------------- | :--------------------------- | :---------------------------- | :-------------------------- |
| Final Train Acc.        | 0.9187            | 0.9690             | 0.9614           |
| Final Train Loss        | 0.2367            | 0.0921             | 0.1112           |
| Test Accuracy           | 0.8746            | 0.9177             | 0.9120           |
| Best Resolution         | 160               | 160                | 160              |
| GPU Util. (%)           | 24.5              | 19.34              | 85.64            |
| CPU Exec (us)           | 20,563            | 31,494             | 15,208           |
| Avg Step Time (us)      | 88,326            | 123,118            | 387.154          |

**Key Observations:**

  * The initial "No Optimization" baseline showed low GPU utilization (24.5%) and a test accuracy of 0.8746.
  * **Resolution Optimization** significantly improved the model's test accuracy to 0.9177. However, its full benefits were observed only when combined with other optimizations (e.g., larger batch sizes, efficient data loading); applying the best resolution in isolation to the unoptimized baseline setup yielded worse results.
  * **Hyperparameter Optimization** achieved the most significant advancement in hardware efficiency, dramatically increasing GPU utilization to 85.64% and substantially reducing CPU execution time to 15,208 us. This successfully addressed the core objective of maximizing hardware utilization and balancing resource usage. Furthermore, this efficiency translated into an improved test accuracy of 0.9120 compared to the initial 10-epoch baseline (0.8746), making it the best overall scenario in terms of balancing performance gains with superior hardware utilization.

## 8\. Key Learnings and Challenges

This project underscored that optimizing AI model training is not solely about achieving the highest accuracy, but critically about training models smartly by considering and adapting to available hardware. Practical skills in identifying bottlenecks, applying targeted software optimizations, and analyzing performance trade-offs were demonstrated.

**Key Learnings:**

  * The effectiveness of resolution tuning is highly context-dependent and amplified by synergistic optimizations.
  * The adoption of large batch sizes and Automatic Mixed Precision (AMP) were key enablers for high GPU saturation.
  * Learning about and observing colleagues' techniques, such as Automatic Mixed Precision (AMP), and gaining preliminary understanding of concepts like model quantization, prefetching, and caching datasets, along with the utility of TensorBoard for profiling, significantly expanded the knowledge base.

**Challenges:**

  * The primary problem encountered was the restricted GPU time on Google Colab, which frequently interrupted training sessions and led to crashes, especially when exploring resource-intensive parameters.
  * Other issues, such as the initial inability to open profiler traces directly on Colab, were minor and largely code-related.

## 9\. Credits

This project was submitted in partial fulfillment of the requirements for the Master in Artificial Intelligence degree at Technische Hochschule Ingolstadt.
