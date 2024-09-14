# **Building a Search Space for SOFTS-ETTh2 Experiment**

This document outlines the reasoning behind the process of constructing the search space for the SOFTS-ETTh2 HyperParameter Optimization (HPO) experiment.

The goal is to develop a hyperparameter tuning pipeline aimed at improving the reported performance of the recently proposed state-of-the-art MLP-based Deep Neural Network, `SOFTS`, on the ETTh2 dataset. To achieve this, we need to consider the following key components:

* Objective Function
* Search Space
* Search Algorithm
* Scheduler (Optional)

The chosen objective function, in line with common practices in hyperparameter tuning literature, is the best validation loss after a certain number of epochs. The training set, validation set, input size, and output size are all consistent with the settings specified in the SOFTS paper (obtained from the official GitHub repository linked to the paper). The reason for reporting the best validation loss, rather than the current validation loss, is that it reflects the peak performance of the evaluated configuration throughout the entire learning process, assuming that the validation dataset distribution aligns (approximately) with the test dataset distribution. If this assumption is invalid, we should consider more robust objective functions, such as those derived from iterative resampling methods like cross-validation. The chosen HPO algorithms are: random search, tpe, bohb (uses hyperband as scheduler) and smac (uses hyperband as scheduler), as explained in the thesis.

To construct the search space, we must first identify the complete list of hyperparameters, select those relevant to our experiment, and assign a probability distribution to each in order to sample them during the hyperparameter optimization process.

The list of most relevant hyperparameters, as provided by the official SOFTS source code, is presented below. These hyperparameters are divided into two categories: global hyperparameters and model-specific hyperparameters. Global hyperparameters are common to most neural network training architectures (at least in a supervised learning context), such as optimizer parameters, batch size, learning rate scheduler type, and the patience of the early stopping callback. Model-specific hyperparameters, on the other hand, pertain to the architecture of the SOFTS model itself, and tuning them requires estimating their effect on model performance (at least approximately). Each of the most relevant hyperparameters from the official SOFTS implementation is analyzed, and a probabilistic distribution is proposed for each one to define the sampling process. 

Probabilistic search spaces, which contain hyperparameters defined by probability distributions, are more expressive and tend to yield better results. They also allow for the specification of conditional dependencies between hyperparameters. However, one must carefully balance complexity with runtime, as more complex search spaces are harder to explore and thus require greater computational resources (e.g., time, distributed systems).

- [Global Hyperparameters](#global-hyperparameters)
   - [Optimizer](#optimizer)
   - [Learning Rate](#learning-rate)
   - [Loss Function](#loss-function)
   - [Batch Size](#batch-size)
   - [Training Epochs](#training-epochs)
- [Model-specific Hyperparameters](#model-specific-hyperparameters)
   - [d_model](#d_model)
   - [d_core](#d_core)
   - [d_ff](#d_ff)
   - [e_layers](#e_layers)
   - [dropout](#dropout)


## **Global Hyperparameters**

### **Optimizer**

The choice of optimizer is critical in training deep neural networks (DNNs) because it directly impacts how the model converges to a solution, how quickly it learns, and how well it generalizes to unseen data. Optimizers dictate how the model’s weights are updated based on the gradients calculated during backpropagation. Their effectiveness can vary depending on the model architecture, dataset, and specific problem. Below, some well-known effects of the optimizer in the training of DNNs are listed:

* **Convergence Speed and Stability**: The optimizer determines how quickly and smoothly a model reaches a minimum in the loss landscape. Some optimizers converge faster (e.g., Adam), while others might have more stable convergence (e.g., SGD with momentum).

* **Avoiding Local Minima**: Optimizers with momentum or adaptive learning rates can help escape local minima and saddle points. This is crucial for high-dimensional loss landscapes with many such points.

* **Generalization**: Some optimizers, such as SGD, are known to offer better generalization because they tend to explore more of the loss landscape, potentially finding flatter minima, which correlate with better generalization.

* **Handling Noise and Sparse Gradients**: Adaptive optimizers (e.g., RMSprop, Adam) can effectively handle noisy gradients or sparse gradients, which are common in certain types of layers (e.g., embeddings in NLP tasks).

Several formal studies and papers investigate the impact of different optimizers on training dynamics, generalization, and convergence:

* **"On the Convergence of Adam and Beyond" (2019)** by Reddi et al.: This paper addresses convergence issues with Adam and proposes a variant called AMSGrad. It delves into how Adam’s adaptive learning rates can sometimes lead to poor convergence in certain situations.

* **"The Marginal Value of Adaptive Gradient Methods in Machine Learning" (2020)** by Wilson et al.: This paper discusses why adaptive methods (like Adam) might generalize worse than simple SGD in some scenarios, despite faster convergence. It highlights how adaptive methods can lead to sharp minima, which correlates with poorer generalization.

* **"Understanding the Role of Momentum in Stochastic Gradient Methods" (2018)** by Sutskever et al.: An older but foundational work that discusses the impact of momentum in speeding up convergence and avoiding local minima.

* **"Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model" (2020)** by Zhang et al.: This paper investigates how different batch sizes affect the performance of optimizers like SGD and Adam, providing insights into which settings are preferable for different scenarios.

These works, among others, illustrate that the choice of optimizer is not universally optimal and should be tuned based on the problem domain, model, and data characteristics. Recent research tends to focus not just on the optimizer itself but also on how it interacts with other hyperparameters, such as batch size and learning rate, to achieve the best balance between convergence speed, stability, and generalization.

Since this experiment focuses primarily on model hyperparameters, and Adam is typically used with default hyperparameters (apart from the learning rate) in this types of models and tasks, we have decided to exclude the optimizer from the tuning process. This allows us to concentrate on the model-specific hyperparameters while also simplifying the search space, as different optimizers come with their own additional hyperparameters.

The initial learning rate for the Adam optimizer will be tuned following common practices. These common practices and standard methodologies also explain the choice of Adam optimizer in the SOFTS paper. 

Quote from SOFTS paper: *We use the ADAM optimizer with an initial learning rate of 3×10−4. This rate is modulated by a cosine learning rate scheduler*

### **Learning Rate**

The learning rate is one of the most crucial hyperparameters in the optimization process of training DNNs. It determines the step size at which the optimizer updates the model's weights during each iteration and significantly affects the model's convergence, stability, and overall performance.

An appropriate learning rate is highly dependent on the architecture, dataset, and problem type. While fixed learning rates are a good starting point, adaptive learning rates, warm-up schedules, and decay strategies are commonly used in practice to balance convergence speed, stability, and generalization. Ongoing theoretical and empirical research continues to refine how learning rates should be tuned for optimal results.

* **Convergence Speed**:
   * **High Learning Rate**: Leads to larger updates to the model’s weights. While this can speed up training, it also risks overshooting the optimal solution, resulting in unstable or non-converging behavior.
   * **Low Learning Rate**: Allows for more fine-grained updates, leading to more stable convergence, but it can slow down the training process considerably, taking much longer to reach a satisfactory solution.
* **Stability**: If the learning rate is too high, the model may exhibit erratic behavior, with the loss oscillating or diverging instead of decreasing. Conversely, a very low learning rate can result in the model getting stuck in local minima or saddle points, as the updates are too small to escape them.
* **Generalization**: The learning rate also affects how well a model generalizes to new, unseen data. A well-chosen learning rate can guide the model toward flatter minima, which often correlates with better generalization performance. Adaptive strategies, such as learning rate schedules or adaptive optimizers, can improve generalization by allowing more aggressive updates early in training and finer adjustments later.
* **Training Time**: A high learning rate may lead to rapid initial reductions in the loss but could make it difficult to find the optimal solution. A low learning rate might guarantee convergence, but at the cost of significantly longer training times.

Several formal studies and papers have explored the role and impact of the learning rate:

* **"Reducing the Learning Rate in SGD: A Cautionary Tale" (2017)** by Loshchilov and Hutter: this paper critiques traditional learning rate decay schedules and introduces alternatives like cosine annealing with warm restarts. It shows that reducing the learning rate too early or too aggressively can harm model performance.

* **"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (2018)** by Keskar et al.: this work discusses the relationship between learning rate, batch size, and generalization. It highlights how large learning rates combined with large batch sizes tend to converge to sharper minima, which can result in worse generalization.

* **"Learning Rate Dropout: Optimal Learning Rates for Deep Networks" (2020)** by Dai et al.: this study proposes dynamically adjusting the learning rate based on training progress, showing how learning rate tuning is key to achieving optimal performance.

* **"Understanding Learning Rate Warmup and Its Effectiveness in Deep Learning" (2019)** by Gotmare et al.: this paper systematically analyzes the impact of learning rate warm-up strategies and shows how they can improve convergence, especially in large-scale models and datasets.

* **"The Marginal Value of Adaptive Gradient Methods in Machine Learning" (2020)** by Wilson et al.: this paper analyzes the effect of adaptive learning rates (like those in Adam) and contrasts them with fixed-rate methods like SGD. It explains how adaptive methods can be more effective in certain settings but may lead to suboptimal generalization in others.

The SOFTS paper suggests a learning rate of 3e-4 for all experiments, which is close to the commonly used default value of 1e-4. However, no information is provided on how this specific value was chosen. Given these considerations, we propose the following log-uniform distribution:

```
learning_rate = loguniform(min=5e-5, max=5e-3)
```

The log-uniform distribution helps improve the coverage of the sampling interval when it spans several orders of magnitude. Its minimum and maximum values are derived after initial testing with random-search-based HPO routines and the value utilized by the authors of SOFTS paper.

Regarding the learning rate scheduler, the official SOFTS implementation uses a cosine learning rate scheduler, defined by the following expression:

$$
\text{lr} = \frac{\text{lr}}{2} \times \left(1 + \cos\left(\frac{\text{epoch}}{n_{epochs}} \times \pi\right)\right)
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

Quote from SOFTS paper: *We use the ADAM optimizer with an initial learning rate of 3×10−4. This rate is modulated by a cosine learning rate scheduler*

### **Loss Function**

In time series forecasting with deep neural networks, the choice of the loss function plays a critical role in determining model performance and behavior. The loss function guides the optimization process, affecting the convergence speed, stability, and generalization capability of the model. Below some well-known effects of the loss function on DNN training are listed:

* **Guiding Model Training:** The loss function quantifies the difference between the model's predictions and the actual target values. During training, the model adjusts its weights to minimize this loss. The form of the loss function directly influences how the model treats different types of errors and adapts its parameters.
* **Error Sensitivity:** Different loss functions treat errors differently. For example, Mean Squared Error (MSE) penalizes larger errors more heavily than smaller ones (due to squaring), making it sensitive to outliers. In contrast, Mean Absolute Error (MAE) treats all errors linearly and is less sensitive to outliers.
* **Forecasting Accuracy and Bias:** The chosen loss function can introduce bias into predictions. For instance, using MSE or MAE tends to optimize models for minimizing point-wise errors but may overlook broader temporal patterns like trends and seasonality. Certain loss functions like the Huber loss combine both MSE and MAE characteristics to balance sensitivity to outliers and overall error minimization.
* **Alignment with Evaluation Metrics:** The effectiveness of a loss function should ideally align with the evaluation metric used for the forecasting task. For example, if the primary evaluation metric is a custom metric like MAPE (Mean Absolute Percentage Error) or SMAPE (Symmetric Mean Absolute Percentage Error), training with a different loss function like MSE might yield suboptimal results.
* **Uncertainty and Distributional Considerations:** Some loss functions are better suited for probabilistic forecasting, where the model outputs a distribution rather than a point estimate. For instance, negative log-likelihood (NLL) or quantile loss functions are commonly used in models that output quantiles or prediction intervals.

Common loss functions in time series forecasting:

* **Mean Squared Error (MSE):** The most common choice, penalizes larger errors more heavily (it is defined a quadratic function of the errors).
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

* **Mean Absolute Error (MAE):** A robust alternative that penalizes all errors linearly (it is defined as a linear function of the errors)
$$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
$$

* **Huber Loss:** Combines MSE and MAE, reducing the impact of outliers.
$$
   \text{Huber Loss} = 
   \begin{cases} 
   \frac{1}{2}(\hat{y}_i - y_i)^2 & \text{for } |\hat{y}_i - y_i| \leq \delta \\
   \delta |\hat{y}_i - y_i| - \frac{1}{2} \delta^2 & \text{otherwise}
   \end{cases}
$$

* **Quantile Loss:** Used in models forecasting quantiles, capturing uncertainty.
   $$
   \text{Quantile Loss} = \max(\tau \cdot (y - \hat{y}), (1 - \tau) \cdot (\hat{y} - y))
   $$
   where $\tau$ is the quantile (e.g., 0.5 for the median).

Several studies provide insights into how different loss functions affect time series forecasting:

* **"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2017)** by Salinas et al. explores using likelihood-based loss functions for probabilistic time series forecasting. This work shows how different loss functions can be used in autoregressive models to handle uncertainty.
* **"Neural Forecasting: Introduction and Literature Overview" (2021)** by Benidis et al. surveys various deep learning methods for forecasting and discusses the impact of loss functions on performance, especially in probabilistic and multivariate settings.
* **"A Comprehensive Review on Neural Network-Based Time Series Forecasting" (2019)** by Zhang et al. discusses the significance of loss functions in neural network forecasting models, comparing different strategies and their impact on forecasting accuracy.
* **"A Comprehensive Review of Loss Functions in Machine Learning and Deep Learning" (2021)** by Prakash et al. provides an analysis of loss functions across domains, including time series forecasting, highlighting the influence of different loss functions on model behavior.

In the SOFTS model, MSE (Mean Squared Error) is used for training, while both MSE and MAE (Mean Absolute Error) are employed for evaluation and benchmarking. To ensure a fair comparison between optimized and default results, we will maintain MSE as the loss function. While optimizing the loss function for a specific objective metric could be explored, this is uncommon in practice. The primary focus of this experiment is on model-specific hyperparameters, which makes this a more valuable contribution due to its novelty and lack of similar experiments.

Quote from SOFTS paper: *The mean squared error (MSE) loss function is utilized for model optimization.*

### **Batch Size**

The effect of batch size when training deep neural networks for time series forecasting is crucial and has been studied extensively in various contexts. Below, some well-known effects of the batch size in the training of DNNs are listed:

* **Convergence Speed**: Smaller batch sizes generally result in noisier updates to model weights, which can improve generalization but slow down convergence. Larger batch sizes typically lead to smoother updates, allowing for faster convergence but can result in overfitting.
* **Generalization Performance**: A medium-sized batch is often preferred in time series forecasting. Extremely small batches might cause the model to be trapped in local minima, while very large batches might cause the model to overfit to the training data.
* **Training Stability**: Larger batch sizes can lead to more stable gradient estimates, but they may require adjustments to learning rates to avoid instability (e.g., using learning rate warm-up techniques).
* **Computational Efficiency**: Larger batch sizes leverage parallelism better on GPUs, leading to faster training times. However, they also demand more memory.
* **Data Dependency**: In time series forecasting, data dependencies and temporal correlations are critical. Small batch sizes can preserve temporal patterns better, whereas large batches may dilute this information.

Several studies have analyzed the impact of batch size in the context of DNNs, though not all are specific to time series forecasting. Key references include:

* **Smith et al. (2018), "Don't Decay the Learning Rate, Increase the Batch Size"**: This paper discusses how increasing the batch size can be an alternative to learning rate decay and can improve generalization performance when coupled with appropriate training strategies.
* **Keskar et al. (2016), "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"**: This study shows that large batch sizes tend to converge to sharp minima in the loss landscape, which can harm generalization, suggesting that small to medium batch sizes are often preferable.
* **LeCun et al. (2012), "Efficient BackProp"**: This classic paper discusses the relationship between batch size, learning rates, and training stability in a broader context.
* **Goodfellow et al. (2016), "Deep Learning"** (Book): This book provides insights into batch size effects in the context of deep neural networks, covering the trade-offs between small and large batch sizes.

While these works do not specifically focus on multivariate time series forecasting, the principles discussed are applicable to this domain. Below, some key Considerations specific to Time Series Forecasting are listed:

* **Temporal Dependency**: For multivariate time series forecasting, capturing the temporal dependencies is crucial. Smaller batches can help preserve these dependencies better by avoiding overly smoothed gradients.
* **Seasonality and Patterns**: When dealing with time series data that exhibit strong seasonality or cyclic behavior, using smaller batch sizes can help models adapt to these patterns better.
* **Memory Constraints**: Given the 2D nature of inputs and outputs in multivariate forecasting, memory usage can become significant. Medium batch sizes (e.g., 32 to 64) are often a good trade-off between performance and resource usage.

As a conclusion, the choice of batch size in multivariate time series forecasting is influenced by the need to balance generalization, convergence speed, and computational efficiency. While small to medium batch sizes (16-64) are commonly used, it's important to experiment with different sizes depending on the dataset, sequence length, and the specific neural network architecture being employed.


In the SOFTS paper, the batch size for the ETTh2 dataset is not specified. However, the source code uses a batch size of 32 multivariate windows, though no explanation is provided for how this value was determined.

Given the impact of batch size on model generalization and the complexity of the optimization landscape, we consider it relevant to include batch size as a hyperparameter in this experiment. Additionally, tuning batch size is recommended in hyperparameter optimization literature. Starting from the default value of 32, we propose the following categorical distribution:

```
batch_size = categorical(values=[8, 16, 32, 64, 128])
```

Several authors suggest that batch sizes should be kept as small as possible, often around 32 samples, which justifies the lower values in the distribution above. Values higher than 128 are excluded to improve the efficiency of the tuning process, as larger batch sizes increase resource consumption and could lead to unstable training due to the more complex optimization landscape.

### **Training Epochs**

The number of training epochs plays a critical role in determining the performance of deep neural networks for time series forecasting, including multivariate forecasting. Below, some well-known effects of the number of training epochs in the training of DNNs are listed:

* **Convergence and Model Performance**: The number of epochs determines how long the model will be trained. Insufficient epochs can lead to underfitting, where the model has not fully learned the underlying patterns in the time series data. Conversely, too many epochs can lead to overfitting, where the model learns noise and specificities in the training data rather than generalizable patterns.
* **Early Stopping**: In practice, instead of relying on a fixed number of epochs, many models employ early stopping. Early stopping monitors the model’s performance on a validation set and stops training when performance starts to degrade, effectively preventing overfitting.
* **Plateauing of Learning**: In time series forecasting, improvements in model performance often plateau after a certain number of epochs. After this point, further training yields diminishing returns or even leads to degradation in performance due to overfitting.
* **Temporal Patterns**: For time series forecasting, where capturing seasonal and cyclical patterns is critical, allowing the model to train for a sufficient number of epochs ensures it can capture these dynamics. However, excessive epochs risk memorizing specific time points instead of general trends.

Some common choices for the number of trainin epochs include:

* **Moderate Epochs (e.g., 50-200 epochs)**: This range is common for many time series forecasting models. The exact number depends on factors like the complexity of the model (e.g., LSTM, CNN, Transformer), the length of the time series, and the amount of available data.
* **Dynamic Epoch Counts**: Often, models do not use a fixed number of epochs. Instead, they combine a relatively high maximum epoch count (e.g., 1000 epochs) with early stopping to end training once performance stabilizes or starts to degrade.Practitioners commonly monitor validation loss or other metrics during training to decide the optimal number of epochs. Early stopping with patience (e.g., waiting for 10 or 20 epochs after the last improvement) is widely used.

And some practical considerations general to DNN training:
* **Complexity of the Model**: Complex models like Transformers or multi-layer LSTMs may require more epochs to converge, while simpler models may need fewer epochs.
* **Dataset Size**: Larger datasets generally require more epochs for the model to fully learn the patterns in the data, while smaller datasets might overfit quickly.

Relevant research and practical guidelines emphasize the importance of combining early stopping with well-chosen epoch limits, making dynamic training durations the most practical approach for multivariate forecasting tasks.

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

**Final Note:** The proposed search space contains 8 hyperparameters, all defined in terms of probability 
distributions and doesn't includes any conditional dependencies between hyperparameters, as these were not necessary to create a sufficiently expressive search space. This is because the current experiment focuses on model-specific hyperparameters, and there are no qualitative hyperparameters whose selection generates other dependent hyperparameters.