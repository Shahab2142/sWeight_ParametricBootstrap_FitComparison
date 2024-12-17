# Extended Likelihood Fitting Project

This project implements and analyzes an **extended likelihood fit** for a toy model that combines signal and background components in two dimensions (\(X\) and \(Y\)). It includes efficient sampling, normalization testing, likelihood fitting, and detailed visualization.

---

## **Overview**
The project evaluates statistical models that describe signal and background processes:
- **Signal Component**: 
   - \(g_s(X)\): Truncated Crystal Ball distribution.
   - \(h_s(Y)\): Truncated exponential distribution.
- **Background Component**:
   - \(g_b(X)\): Uniform distribution.
   - \(h_b(Y)\): Truncated normal distribution.

The workflow includes:
1. Generating samples from these distributions.
2. Verifying their normalization.
3. Performing an extended likelihood fit to retrieve parameters.
4. Visualizing results for validation.
5. Conducting **bootstrap studies** to assess bias and uncertainty in parameter estimation.

---

## **Directory Structure**

The repository consists of the following files:

### **1. `distributions_sampling_utils.py`**
- Implements the signal and background probability distributions in both \(X\) and \(Y\).
- Contains functions for:
   - Truncated Crystal Ball PDF (`g_s(X)`).
   - Exponential PDF (`h_s(Y)`).
   - Truncated Normal PDF (`h_b(Y)`).
   - Uniform PDF (`g_b(X)`).
   - Sampling from these distributions.
- Key Functions:
   - `crystal_ball_pdf_vectorized`: Vectorized Crystal Ball PDF.
   - `sample_componentwise`: Efficient sampling from signal/background components.
   - `test_normalization`: Verifies normalization of distributions numerically.

---

### **2. `fits_util.py`**
- Implements the **extended likelihood fit** using `iminuit`.
- Defines the negative log-likelihood for fitting the joint PDF \(f(X, Y)\), which combines signal and background contributions.
- Key Function:
   - `perform_fit`: Performs the likelihood fit with parameter bounds and initial guesses.

---

### **3. `plots_utils.py`**
- Provides utilities for visualizing both theoretical and sampled distributions.
- Key Functions:
   - `plot_distributions`: Visualizes the marginal and joint distributions.
   - `plot_sampled_distributions`: Compares sampled distributions with theoretical marginals.

---

### **4. Notebook (Main Workflow)**
The notebook demonstrates the end-to-end analysis:
1. **Normalization Testing**:
   - Verifies that individual and joint distributions integrate to 1.
2. **Visualization**:
   - Plots theoretical PDFs and compares them with sampled data.
3. **Extended Likelihood Fit**:
   - Performs the fit on generated samples and retrieves parameters.
4. **Benchmarking**:
   - Measures execution time for sampling and fitting routines.
5. **Bootstrap Study**:
   - Evaluates bias and uncertainty of the decay parameter (\(\lambda\)) as a function of sample size.
6. **Weighted Fits with s-Weights**:
   - Implements a two-step procedure:
      - Step 1: Fit in \(X\) to compute **s-weights** (signal probability for each sample).
      - Step 2: Perform a weighted fit in \(Y\) using these s-weights.

---

## **Dependencies**
- **Python**: 3.8 or higher
- **NumPy**: Array operations and random sampling.
- **SciPy**: Truncated normal sampling and numerical integration.
- **Matplotlib**: Visualization of results.
- **Seaborn**: Enhanced plotting aesthetics.
- **iminuit**: Minimization of the negative log-likelihood.
- **Numba**: JIT compilation for efficient numerical operations.

---

## **How to Run**

1. **Install Dependencies**:
   ```bash
   pip install numpy scipy matplotlib seaborn iminuit numba
