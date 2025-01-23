# **Extended Likelihood Fitting for Signal and Background Separation**

## **Project Overview**
This project models a two-dimensional probability distribution $f(X, Y)$ that describes **signal** and **background** processes using a combination of analytically defined probability density functions (PDFs). The main objectives are:
1. **Sample Generation**: Generate high-statistics samples from the defined signal and background distributions.
2. **Normalization Verification**: Validate that all PDFs and their joint distribution integrate correctly over their respective domains.
3. **Likelihood Fitting**: Perform an **extended likelihood fit** using the generated samples to recover model parameters.
4. **Statistical Analysis**:
   - Perform **parametric bootstrapping** to evaluate the bias and uncertainty of the model parameters.
   - Implement **weighted fits** using **s-weights** to project signal information into the $Y$-dimension after fitting in $X$.

---

## **Mathematical Definition of the Model**

The total probability density function $f(X, Y)$ is a mixture of **signal** and **background** components:
$$
f(X, Y) = f \cdot s(X, Y) + (1 - f) \cdot b(X, Y)
$$
where:
- $f$: Fraction of the total density attributed to the **signal**.
- $s(X, Y)$: Signal joint PDF, which factorizes as:
  $s(X, Y) = g_s(X) \cdot h_s(Y)$
- $b(X, Y)$: Background joint PDF, which factorizes as:
  $b(X, Y) = g_b(X) \cdot h_b(Y)$

---

### **Signal PDFs**
1. **$g_s(X)$**: The signal distribution in $X$ follows a **Crystal Ball** function, which is a blend of a Gaussian and a power law tail. It is expressed as:
   $$
   g_s(X) = 
   \begin{cases} 
   e^{-\frac{Z^2}{2}} & \text{for } Z \geq -\beta \\
   \left( \frac{m}{\beta} \right)^m e^{-\frac{\beta^2}{2}} \left( \frac{m}{\beta} - \beta - Z \right)^{-m} & \text{for } Z < -\beta
   \end{cases}
   $$
   where $Z = \frac{X - \mu}{\sigma}$, and $\mu$, $\sigma$, and $\beta$ are parameters defining the mean, standard deviation, and the tail exponent, respectively.

2. **$h_s(Y)$**: The signal distribution in $Y$ follows a truncated **exponential distribution**:
   $$
   h_s(Y) = \lambda_s e^{-\lambda_s Y}, \quad \text{for } Y \in [0, 10]
   $$
   where $\lambda_s$ is the decay constant.

---

### **Background PDFs**
1. **$g_b(X)$**: The background distribution in $X$ is uniform over the interval $[0, 5]$:
   $$
   g_b(X) = \frac{1}{5}, \quad \text{for } X \in [0, 5]
   $$
2. **$h_b(Y)$**: The background distribution in $Y$ follows a truncated **normal distribution** with mean $\mu_b$ and standard deviation $\sigma_b$:
   $$
   h_b(Y) = \frac{1}{\sigma_b \sqrt{2\pi}} e^{-\frac{(Y - \mu_b)^2}{2\sigma_b^2}}, \quad \text{for } Y \in [0, 10]
   $$

---

## **Key Components of the Project**

### **1. `distributions_sampling_utils.py`**
This file defines all the signal and background PDFs, along with functions for sampling from these distributions.

- **Functions**:
   - `crystal_ball_pdf_vectorized`: Computes the Crystal Ball PDF.
   - `h_s_vectorized`: Computes the truncated exponential PDF.
   - `h_b_vectorized`: Computes the truncated normal PDF.
   - `g_b_vectorized`: Computes the uniform PDF.
   - `sample_componentwise`: Generates samples from $f(X, Y)$ by probabilistically assigning each sample to the signal or background components.

- **Normalization Check**:
   - `test_normalization`: Numerically integrates the PDFs to verify they are correctly normalized over their domains.

---

### **2. `fits_util.py`**
This file performs the **extended likelihood fit** using the **`iminuit`** library.

- **Extended Likelihood Fit**:
   The extended likelihood includes:
   - A **Poisson term** to account for fluctuations in the total sample size.
   - The **log-sum** of the joint PDF values over all samples.

   The negative log-likelihood to be minimized is:
   $$
   \mathcal{L} = -N_{\text{expected}} + N_{\text{observed}} \log(N_{\text{expected}}) + \sum_{i} \log f(X_i, Y_i)
   $$

- **Key Function**:
   - `perform_fit`: Performs the extended likelihood fit using the observed samples, parameter bounds, and initial guesses.

---

### **3. `plots_utils.py`**
This file provides functions for **visualizing** the theoretical distributions, sampled data, and the results of the fits.

- **Functions**:
   - `plot_distributions`: Plots the theoretical marginal distributions $f_X(X)$, $f_Y(Y)$, and the joint distribution $f(X, Y)$.
   - `plot_sampled_distributions`: Compares sampled histograms to the theoretical marginal distributions.

---

## **Notebook Workflow**
The notebook demonstrates the complete analysis pipeline:

1. **Normalization Verification**:
   - Checks that all component PDFs and the joint PDF integrate to 1.

2. **Sample Generation**:
   - Generates $N = 100,000$ samples from $f(X, Y)$.

3. **Extended Likelihood Fit**:
   - Fits the generated data to recover the true model parameters.
   - Validates the fit convergence and uncertainties.

4. **Statistical Analysis**:
   - **Benchmarking**: Measures the execution time for sampling and fitting routines.
   - **Parametric Bootstrapping**:
     - Repeats the sample generation and fitting process for varying sample sizes (500, 1000, 2500, 5000, and 10000).
     - Evaluates bias and uncertainty in the decay parameter $\lambda_s$.
   - **s-Weights Analysis**:
     - Performs a **fit in $X$** to calculate s-weights (signal probabilities for each sample).
     - Uses the s-weights to perform a weighted fit in $Y$.

---

## **Mathematical Background of s-Weights**
The s-weights method projects signal contributions into the $Y$-dimension:
1. Perform a fit in $X$ to estimate the signal fraction and calculate s-weights for each sample:
   $$
   w_i = \frac{f \cdot g_s(X_i)}{f \cdot g_s(X_i) + (1 - f) \cdot g_b(X_i)}
   $$
2. Use the s-weights as weights in a weighted likelihood fit to estimate the decay constant $\lambda_s$ in $Y$.

---

## **Dependencies**
To run this project, you will need the following libraries:
- **NumPy**: Efficient numerical computations.
- **SciPy**: Integration and truncated normal sampling.
- **Numba**: Just-in-time compilation for efficient sampling.
- **iminuit**: Minimization of the negative log-likelihood.
- **Matplotlib** and **Seaborn**: Visualization.

---

## **License**

This project is licensed under the MIT License.

---

## **Declaration of Autogenerative Tools**

In this project, I used GitHub Copilot as a coding assistant. I mainly used it to generate the plots in part c), the table in d), and the bias and variance plots in part e) and f). I also used it within my code to create comments, docstring my functions, debug and format markdown cells explaining the different sections in the notebook.

In regards to the report, I used the Overleaf AI which uses ChatGPT to help write the mathematics in LaTeX format, and also used it to correctly format various sections of the report as I am quite new to LaTeX.
