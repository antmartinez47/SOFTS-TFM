# **Building a Search Space for the SOFTS-ETTh2 HPO Experiment**

This document outlines the reasoning behind the process of constructing the search space for the SOFTS-ETTh2 HyperParameter Optimization (HPO) experiment.

The goal is to develop a hyperparameter tuning pipeline aimed at improving the reported performance of the recently proposed state-of-the-art MLP-based Deep Neural Network, `SOFTS`, on the ETTh2 dataset. To achieve this, we need to consider the following key components:

* Objective Function
* Search Space
* Search Algorithm
* Scheduler (Optional)

The chosen objective function, in line with common practices in hyperparameter tuning literature, is the best validation loss after a certain number of epochs. The training set, validation set, input size, and output size are all consistent with the settings specified in the SOFTS paper (obtained from the official GitHub repository linked to the paper). The reason for reporting the best validation loss, rather than the current validation loss, is that it reflects the peak performance of the evaluated configuration throughout the entire learning process, assuming that the validation dataset distribution aligns (approximately) with the test dataset distribution. If this assumption is invalid, we should consider more robust objective functions, such as those derived from iterative resampling methods like cross-validation.

To construct the search space, we must first identify the complete list of hyperparameters, select those relevant to our experiment, and assign a probability distribution to each in order to sample them during the hyperparameter optimization process.

The full list of hyperparameters, as provided by the official SOFTS source code, is presented below. These hyperparameters are divided into two categories: global hyperparameters and model-specific hyperparameters. Global hyperparameters are common to most neural network training architectures (at least in a supervised learning context), such as optimizer parameters, batch size, learning rate scheduler type, and the patience of the early stopping callback. Model-specific hyperparameters, on the other hand, pertain to the architecture of the SOFTS model itself, and tuning them requires estimating their effect on model performance (at least approximately). Each of the most relevant hyperparameters from the official SOFTS implementation is analyzed, and a probabilistic distribution is proposed for each one to define the sampling process. 

Probabilistic search spaces, which contain hyperparameters defined by probability distributions, are more expressive and tend to yield better results. They also allow for the specification of conditional dependencies between hyperparameters. However, one must carefully balance complexity with runtime, as more complex search spaces are harder to explore and thus require greater computational resources (e.g., time, distributed systems).

## **Global Hyperparameters**

### **Optimizer**

The choice of optimizer is critical in training deep neural networks (DNNs) because it directly impacts how the model converges to a solution, how quickly it learns, and how well it generalizes to unseen data. Optimizers dictate how the model’s weights are updated based on the gradients calculated during backpropagation. Their effectiveness can vary depending on the model architecture, dataset, and specific problem.

Since this work focuses primarily on model hyperparameters, and Adam is typically used with default hyperparameters (apart from the learning rate), we have decided to exclude the optimizer from the tuning process. This allows us to concentrate on the model-specific hyperparameters while also simplifying the search space, as different optimizers come with their own additional hyperparameters.

The initial learning rate for the Adam optimizer will be tuned following common practices. These common practices and standard methodologies also inform the choice of optimizer in the SOFTS paper. 

### **Learning Rate and Learning Rate Schedulers**

The learning rate is one of the most crucial hyperparameters in the optimization process of training DNNs. It determines the step size at which the optimizer updates the model's weights during each iteration and significantly affects the model's convergence, stability, and overall performance.

An appropriate learning rate is highly dependent on the architecture, dataset, and problem type. While fixed learning rates are a good starting point, adaptive learning rates, warm-up schedules, and decay strategies are commonly used in practice to balance convergence speed, stability, and generalization. Ongoing theoretical and empirical research continues to refine how learning rates should be tuned for optimal results.

The SOFTS paper suggests a learning rate of 3e-4 for all experiments, which is close to the commonly used default value of 1e-4. However, no information is provided on how this specific value was chosen. Given these considerations, we propose the following log-uniform distribution:

```
learning_rate = loguniform(min=5e-5, max=5e-3)
```

The log-uniform distribution helps improve the coverage of the sampling interval when it spans several orders of magnitude. Its minimum and maximum values are derived after initial testing with random-search-based HPO routines and the value utilized by the authors of SOFTS paper.

Regarding the learning rate scheduler, the official SOFTS implementation uses a cosine learning rate scheduler, defined by the following expression:

$$
\text{lr} = \frac{\text{args.learning\_rate}}{2} \times \left(1 + \cos\left(\frac{\text{epoch}}{\text{args.train\_epochs}} \times \pi\right)\right)
$$

This works as follows:

* The learning rate starts at its maximum value
* It then decreases according to a cosine curve, reaching its minimum towards the end of training.
* This scheduler is widely used because it provides a smooth reduction in learning rate, which can result in better convergence in some cases.

The cosine function oscillates between -1 and 1 over a full cycle of $\pi$, but here it is shifted to oscillate between 0 and 1 by adding 1 and dividing by 2. Consequently, the learning rate starts high, gradually decreases, and stabilizes at the end of training. The scheduler is updated at the end of each epoch. 

Additionally, the SOFTS repository includes other learning rate schedulers, such as a stepwise exponential decay scheduler (referred to as `type1`) and a fixed-scheme scheduler (referred to as `type2`). Given the significant impact of learning rate schedules on the model's learning capabilities, this has been deemed a relevant tunable hyperparameter. Moreover, including it in the experiment allows for performance comparisons between different learning rate schedulers, ranging from more complex schemes, like the cosine-based scheduler, to simpler ones, like exponential decay. Follwoing this dicussion, a categorical distribution over the values {cosine, type1} is proposed for the choice of learning rate scheduler:

```
lradj = categorical(values=['cosine', 'type1'])
```

### **Loss Function**

In time series forecasting using deep neural networks, the choice of loss function is critical for guiding the optimization process and influencing the model's performance, convergence speed, stability, and generalization capability.

In the SOFTS model, MSE (Mean Squared Error) is used for training, while both MSE and MAE (Mean Absolute Error) are employed for evaluation and benchmarking. To ensure a fair comparison between optimized and default results, we will maintain MSE as the loss function. While optimizing the loss function for a specific objective metric could be explored, this is uncommon in practice. The primary focus of this experiment is on model-specific hyperparameters, which makes this a more valuable contribution due to its novelty and lack of similar experiments.

### **Batch Size**

The choice of batch size affects the balance between generalization, convergence speed, and computational efficiency. While small to medium batch sizes (16–64) are often used, experimenting with different sizes is important, depending on the dataset, sequence length, and specific neural network architecture being applied. Adjustments in learning rates and other hyperparameters are typically necessary when modifying batch sizes.

In the SOFTS paper, the batch size for the ETTh2 dataset is not specified. However, the source code uses a batch size of 32 multivariate windows, though no explanation is provided for how this value was determined.

Given the impact of batch size on model generalization and the complexity of the optimization landscape, we consider it relevant to include batch size as a hyperparameter in this experiment. Additionally, tuning batch size is recommended in hyperparameter optimization literature. Starting from the default value of 32, we propose the following categorical distribution:

```
batch_size = categorical(values=[8, 16, 32, 64, 128])
```

Several authors suggest that batch sizes should be kept as small as possible, often around 32 samples, which justifies the lower values in the distribution above. Values higher than 128 are excluded to improve the efficiency of the tuning process, as larger batch sizes increase resource consumption and could lead to unstable training due to the more complex optimization landscape.

### **Number of Epochs and Early Stopping Callback**

The number of training epochs is a critical factor in determining the performance of deep neural networks, particularly in time series forecasting. Below is an overview of how the number of epochs influences model performance, common practices, and relevant studies.

In this experiment, the number of training epochs plays a crucial role, especially in Multi-Fidelity Optimization methods like BOHB or SMAC. Depending on the search algorithm used, the role of epochs differs. For non-multi-fidelity algorithms, such as Random Search or Hyperopt TPE, the number of epochs can be tuned. However, for multi-fidelity algorithms, the number of epochs defines the maximum budget, as we consider "budget" to represent the number of epochs (iterations of the training algorithm). Therefore, it becomes a parameter of the scheduler in the tuning process. 

Given this distinction, and in the interest of evaluating a broader range of configurations (yielding more informed results), we will avoid tuning the number of epochs. Focusing on more configurations with fewer epochs provides more value than fewer configurations with many epochs, which could be suboptimal due to time constraints and hardware limitations. The behavior of this hyperparameter will vary depending on the algorithm used:

* For non-multi-fidelity methods (random search, tpe), the value will be set to 8 training epochs (default is 20 epochs), and an Early Stopping callback will be applied with a patience of 3, as used in the official source code for the ETTh2 dataset. This approach will enhance the efficiency of the optimization process. The choice of 8 epochs is based on the learning trajectories observed for the ETTh2 dataset, where the actual number of epochs after early stopping for each horizon value ranges between 2 and 8.
  
* For multi-fidelity methods (bohb, smac), the value of 8 training epochs will be assigned to the `max_budget` parameter of the corresponding scheduler (e.g., Hyperband or Successive Halving), and no early stopping callback will be used, as stopping mechanisms are handled by the scheduler.

## **Model-Specific Hyperparameters**

### **`d_model`**

This hyperparameter defines the dimensionality of the model's internal layers, particularly the embedding and projection layers. It typically refers to the number of features (or channels) in the embedding representation, which is passed through various components of the network.

In SOFTS, `d_model` serves as the dimension of the initial embedding (obtained via a linear projection) over the time series, which is then fed into the STAR blocks. Within each STAR block, the embedding is processed to form the core representation (`d_core`). A repeat and concatenation operation then produces a tensor of dimensions `num_channels × (d_model + d_core)`. Finally, a GELU-MLP projection recovers the dimensionality of `d_model`, and a residual connection is added (this constitutes the output of the STAR block).

The SOFTS paper reports experimenting with `d_model` values of {128, 256, 512}, with 128 being used for the ETTh2 dataset across all horizons. This value will serve as the default for our tuning process.

To provide a more expressive and complex search space, we propose extending the range specified in the paper in both directions. A categorical distribution with the following values will be used: $\{32, 64, 128, 256, 512\}$. Values below 32 are excluded to avoid underfitting, while values above 512 are avoided due to the high computational cost during training.

```
d_model = categorical(values=[32, 64, 128, 256, 512])
```

### **`d_core`**

In SOFTS, `d_core` defines the dimensionality of the core representation, the central processing unit within the STAR block. For the ETTh2 dataset, the default value is 64 for all horizons. While the authors experimented with values in the range {64, 128, 256, 512}, they found that performance was similar across all values. This could be due to the intrinsic low impact of this hyperparameter at higher values or the small set of candidates tested. However, since the ETTh2 dataset was not included in the experiment, the conclusions may not fully apply here.

Due to its conceptual similarity with `d_model`, we opt to use the same distribution:

```
d_core = categorical(values=[32, 64, 128, 256, 512])
```

### **`d_ff`**

In transformer models, `d_ff` generally refers to the dimension of the intermediate layer in the feedforward network that follows the multi-head attention layers. It is typically larger than `d_model` to allow for richer transformations before reducing the dimensionality back to `d_model`.

In SOFTS, `d_ff` represents the hidden dimension of a two-layered CNN with dropout and GELU activation, added to the output of the STAR block.

The source code suggests $d_{ff} = 4 \cdot d_{model}$, a common choice in transformer models. However, the official SOFTS implementation uses $d_{ff} = d_{model}$, likely to reduce computational complexity.

Since `d_ff` is usually dependent on `d_model`, and the paper does not provide specific information about its value in the experiments, we will define a candidate distribution that scales proportionally with `d_model`:

$$ d_{ff} = \alpha \cdot d_{model} $$

Where $\alpha$ is a hyperparameter that will be tuned. Typical values for $\alpha$ range from 2 to 8, depending on the model architecture. For this experiment, we propose the following categorical distribution for $\alpha$:

```
alpha = categorical(values=[1, 2, 3, 4])
```

### **`e_layers`**

This hyperparameter controls the number of encoder layers used in the model, specifically how many STAR blocks are stacked sequentially. The STAR block is responsible for key operations such as core representation, aggregation, and fusion.

The default value for the ETTh2 dataset is 2 for all horizons. The authors experimented with values in the range {1, 2, 3, 4} and found that higher values are beneficial for more complex datasets. However, since ETTh2 was not included in their experiment, we will use the same distribution:

```
e_layers = categorical(values=[1, 2, 3, 4])
```

### **`dropout`**

The dropout rate (dropout) determines the probability of dropping neurons in the dropout layers, which are used throughout the SOFTS model. Dropout is a regularization technique employed to prevent overfitting by randomly setting a fraction of the neurons to zero during each forward pass. This helps the model become less reliant on specific neurons and improves generalization.

Although the SOFTS paper does not explicitly mention dropout, inspection of the code reveals that dropout is implemented but not applied for the ETTh2 dataset: it is applied in the CNN submodule that comes the output of the MLP fusion block within the STAR module (see Figure 1 of SOFTS-TFM README file), just before the residual connection. In particular, dropout is applied after each of the two pointwise convoluitional layers that compose the CNN submodule. Given the widespread use of dropout to improve performance in complex and noisy environments (such as ETTh2), it has been deemed a good candidate for tuning. We propose the following log-uniform distribution:

```
dropout = loguniform(min=8e-4, max=1.2e-1)
```

The minimum and maximum values were derived based on common practices for CNN-based DNNs, as well as initial tests conducted using random-search-based HPO routines.

**Final Note:** Resulting search space contains 8 hyperparameters, all defined in terms of probability 
distributions. Furthermore, the proposed search space does not include any conditional dependencies between hyperparameters, as these were not necessary to create a sufficiently expressive search space. This is because the current experiment focuses on model-specific hyperparameters, and there are no qualitative hyperparameters whose selection generates other dependent hyperparameters.