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

[cite\_start]Training large AI models presents significant challenges due to hardware constraints, often leading to prolonged training times, out-of-memory errors, and inefficient hardware utilization[cite: 34]. [cite\_start]This project addresses these issues by focusing on optimizing AI model training for faster experimentation, reduced energy consumption, and maximum utilization of existing computational infrastructure, particularly on limited resources[cite: 35].

[cite\_start]We used the Intel Image Classification dataset, which consists of RGB JPEG images across six categories, initially standardized to $150 \\times 150$ pixels[cite: 36]. [cite\_start]For the deep learning model, SqueezeNet version 1.1 was selected due to its balance of accuracy (0.89) and compact size (4.8 MB), which aligned well with the project's optimization goals under hardware limitations[cite: 36, 37]. [cite\_start]The primary development and experimentation environment was Google Colab's T4 GPU, which offered crucial GPU access despite daily usage limits[cite: 38, 39].

## 2\. Project Goals

The main objectives of this project were to:

  * [cite\_start]Achieve stable training and improved accuracy for the SqueezeNet 1.1 model[cite: 42].
  * [cite\_start]Maximize CPU utilization concurrently, aiming for balanced hardware usage, as initial observations via TensorBoard showed significant CPU underutilization (around 20%)[cite: 43].
  * [cite\_start]Develop real-world, hardware-aware AI engineering skills[cite: 44].
  * [cite\_start]Implement and integrate hyperparameter optimization and resolution tuning into a combined pipeline[cite: 45].

## 3\. Methods and Technologies Used

[cite\_start]To address training instability, suboptimal accuracy, and inefficient hardware utilization, we employed a suite of software optimization techniques[cite: 48].

### Deep Learning Framework

  * [cite\_start]**PyTorch**: The project was implemented in Python within the Google Colab environment, leveraging the PyTorch framework[cite: 62].

### Optimization Framework

  * [cite\_start]**Optuna**: Chosen as an open-source hyperparameter optimization framework for its efficient, automated search capabilities[cite: 49]. [cite\_start]Optuna explores a search space of hyperparameters through "trials," defining an "objective function" to maximize (e.g., validation accuracy)[cite: 50]. [cite\_start]This approach proved superior to manual tuning, grid search, or random search due to its adaptiveness and efficiency, which was crucial given Google Colab's GPU limitations[cite: 51, 52].
      * [cite\_start]**Tree-structured Parzen Estimator (TPE) Sampler**: Both optimization studies utilized Optuna's TPE sampler[cite: 53]. [cite\_start]TPE is a Bayesian optimization algorithm that builds probability distributions based on past evaluated trials to suggest new hyperparameters more likely to improve the objective[cite: 54, 55]. [cite\_start]This directed search leads to faster convergence to optimal solutions, directly aligning with the goal of faster experimentation and maximizing the utility of existing computational infrastructure[cite: 56].

### Optimization Techniques Implemented

  * [cite\_start]**Hyperparameter Optimization** (`optuna_optimizer.py`): Using the `OptunaTuner` class, validation accuracy was maximized by tuning the learning rate, batch size, and epochs[cite: 58, 69]. [cite\_start]The batch size search space was expanded to include larger values[cite: 540].
  * [cite\_start]**Resolution Tuning** (`resolution_tuner.py`): The `ResolutionTuner` class dynamically optimized image resolution (e.g., [144, 160, 176, 192]), also aiming to maximize validation accuracy[cite: 59, 70, 541].
  * [cite\_start]**Automatic Mixed Precision (AMP)**: Implemented within `trainer_2.py` using `torch.amp.autocast` and `torch.cuda.amp.GradScaler`[cite: 67]. [cite\_start]AMP improves training speed and reduces memory footprint by using lower precision operations and managing gradient scaling[cite: 68]. [cite\_start]It was a key enabler for achieving high GPU saturation[cite: 112].
  * [cite\_start]**Learning Rate Scheduler (CosineAnnealingLR)**: Employed to adjust the learning rate during training[cite: 537, 538, 540, 541].
  * [cite\_start]**DataLoader with `num_workers` and `pin_memory=True`**: Used for efficient data loading, contributing to maximum CPU utilization[cite: 79, 39, 112]. [cite\_start]`pin_memory=True` was added for DataLoaders[cite: 540, 541].
  * [cite\_start]**Adam Optimizer**: Used as the primary optimizer for training the model[cite: 540, 541].
  * [cite\_start]**SqueezeNet1\_1\_Weights.DEFAULT**: Utilized for initializing the SqueezeNet model with pre-trained weights[cite: 540].

### Performance Monitoring

  * [cite\_start]**TorchProfiler** (`profiler.py`): The `TorchProfiler` class captured detailed CPU and CUDA activity using `torch.profiler` and monitored system resources via `psutil`[cite: 71, 539]. This provided crucial insights into hardware utilization.
  * [cite\_start]**TensorBoard**: Used for profiling, as mentioned in the key learnings, and for visualizing training results[cite: 117].

### Data Handling

  * [cite\_start]**`dataset.py`**: Handled dataset downloading via `kagglehub` and applied image transformations, including dynamic resizing[cite: 66, 537].
  * [cite\_start]**`utility.py`**: Provided helper functions for inference and evaluation[cite: 72, 538].

## 4\. Project Structure

The project is organized into a modular structure to ensure maintainability and clarity.

```
your_project_folder/
├── modules/
[cite_start]│   ├── dataset.py            # Handles Intel Image Classification dataset loading and transformations. [cite: 537]
[cite_start]│   ├── trainer_2.py          # Custom Trainer class with AMP, LR scheduler, etc. [cite: 538]
[cite_start]│   ├── utility.py            # Helper functions for inference, evaluation, plotting. [cite: 539]
[cite_start]│   └── profiler.py           # TorchProfiler for performance monitoring. [cite: 540]
[cite_start]├── optuna_optimizer.py       # OptunaTuner class for hyperparameter optimization. [cite: 540]
[cite_start]├── resolution_tuner.py       # ResolutionTuner class for image resolution optimization. [cite: 541]
[cite_start]├── requirements.txt          # Lists all necessary Python packages. [cite: 542]
└── main_pipeline.ipynb       # Main Jupyter Notebook to run scenarios.
```

**Key Points:**

  * [cite\_start]The `modules` folder contains core utility classes[cite: 544].
  * [cite\_start]`optuna_optimizer.py`, `resolution_tuner.py`, and `requirements.txt` must be in the **same root directory as your `modules` folder** and your main execution script (e.g., `main_pipeline.ipynb`), but *not inside* the `modules` folder itself[cite: 545].
  * [cite\_start]The folder named `others` (not explicitly shown in structure, but mentioned in source) contains the saved model and profiler traces[cite: 546].

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

      * [cite\_start]`ipython` [cite: 537]
      * [cite\_start]`ipywidgets` [cite: 537]
      * [cite\_start]`kagglehub` [cite: 537]
      * [cite\_start]`plotly` [cite: 537]
      * [cite\_start]`torch` [cite: 537]
      * [cite\_start]`torchmetrics` [cite: 537]
      * [cite\_start]`torchvision` [cite: 537]

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

[cite\_start]The project compared the performance of SqueezeNet 1.1 across three distinct scenarios[cite: 87]:

| Metric / Scenario       | [cite\_start]Baseline (10 Ep) [cite: 96] | Res. Opt. (10 Ep) [cite\_start][cite: 96] | HP Opt. (20 Ep) [cite\_start][cite: 96] |
| :---------------------- | :--------------------------- | :---------------------------- | :-------------------------- |
| Final Train Acc.        | [cite\_start]0.9187 [cite: 96]            | [cite\_start]0.9690 [cite: 96]             | [cite\_start]0.9614 [cite: 96]           |
| Final Train Loss        | [cite\_start]0.2367 [cite: 96]            | [cite\_start]0.0921 [cite: 96]             | [cite\_start]0.1112 [cite: 96]           |
| Test Accuracy           | [cite\_start]0.8746 [cite: 96]            | [cite\_start]0.9177 [cite: 96]             | [cite\_start]0.9120 [cite: 96]           |
| Best Resolution         | [cite\_start]160 [cite: 96]               | [cite\_start]160 [cite: 96]                | [cite\_start]160 [cite: 96]              |
| GPU Util. (%)           | [cite\_start]24.5 [cite: 96]              | [cite\_start]19.34 [cite: 96]              | [cite\_start]85.64 [cite: 96]            |
| CPU Exec (us)           | [cite\_start]20,563 [cite: 96]            | [cite\_start]31,494 [cite: 96]             | [cite\_start]15,208 [cite: 96]           |
| Avg Step Time (us)      | [cite\_start]88,326 [cite: 96]            | [cite\_start]123,118 [cite: 96]            | [cite\_start]387.154 [cite: 96]          |

**Key Observations:**

  * [cite\_start]The initial "No Optimization" baseline showed low GPU utilization (24.5%) and a test accuracy of 0.8746[cite: 102].
  * [cite\_start]**Resolution Optimization** significantly improved the model's test accuracy to 0.9177[cite: 103]. [cite\_start]However, its full benefits were observed only when combined with other optimizations (e.g., larger batch sizes, efficient data loading); applying the best resolution in isolation to the unoptimized baseline setup yielded worse results[cite: 105, 106].
  * [cite\_start]**Hyperparameter Optimization** achieved the most significant advancement in hardware efficiency [cite: 108][cite\_start], dramatically increasing GPU utilization to 85.64% and substantially reducing CPU execution time to 15,208 us[cite: 109]. [cite\_start]This successfully addressed the core objective of maximizing hardware utilization and balancing resource usage[cite: 110]. [cite\_start]Furthermore, this efficiency translated into an improved test accuracy of 0.9120 compared to the initial 10-epoch baseline (0.8746), making it the best overall scenario in terms of balancing performance gains with superior hardware utilization[cite: 111].

## 8\. Key Learnings and Challenges

[cite\_start]This project underscored that optimizing AI model training is not solely about achieving the highest accuracy, but critically about training models smartly by considering and adapting to available hardware[cite: 120]. [cite\_start]Practical skills in identifying bottlenecks, applying targeted software optimizations, and analyzing performance trade-offs were demonstrated[cite: 121].

**Key Learnings:**

  * [cite\_start]The effectiveness of resolution tuning is highly context-dependent and amplified by synergistic optimizations[cite: 107].
  * [cite\_start]The adoption of large batch sizes and Automatic Mixed Precision (AMP) were key enablers for high GPU saturation[cite: 112].
  * [cite\_start]Learning about and observing colleagues' techniques, such as Automatic Mixed Precision (AMP), and gaining preliminary understanding of concepts like model quantization, prefetching, and caching datasets, along with the utility of TensorBoard for profiling, significantly expanded the knowledge base[cite: 117].

**Challenges:**

  * [cite\_start]The primary problem encountered was the restricted GPU time on Google Colab, which frequently interrupted training sessions and led to crashes, especially when exploring resource-intensive parameters[cite: 118].
  * [cite\_start]Other issues, such as the initial inability to open profiler traces directly on Colab, were minor and largely code-related[cite: 119].

## 9\. Credits

[cite\_start]This project was submitted in partial fulfillment of the requirements for the Master in Artificial Intelligence degree [cite: 6, 7, 8, 9] [cite\_start]at Technische Hochschule Ingolstadt[cite: 5, 19].

