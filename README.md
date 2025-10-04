# Robust Bayesian Analysis of Extreme Value Laws

**Date:** April 7, 2025  
**Authors:** Pablo Soto Martín, Tiziano Fassina  

This project presents a **robust Bayesian analysis** applied to **extreme value distributions**, combining **theoretical results** and **numerical methods** with practical applications in **hydrology** and **meteorology**.

---

## **1. Background and Motivation**
- Exact specification of **prior distributions** in extreme value analysis is challenging.  
- **Expert judgments** are often uncertain and rarely directly available.  
- **Robust Bayesian methods** allow handling this uncertainty by considering **classes of prior distributions** instead of a single one.

---

## **2. Theoretical Approach**
- **Formalization using the Generalized Extreme Value (GEV) distribution**:  
  - Parameters: µ ∈ ℝ, σ > 0, ξ ∈ ℝ  
  - Standard CDF and PDF define the likelihood for observed data.  
- **Robust analysis**:  
  - Evaluate the sensitivity of posterior quantities with respect to different priors.  
  - Use **generalized moment constraints** to define an admissible class Γ of priors.  
  - Robustness is measured by the **range of posterior quantiles**.

---

## **3. Numerical Method**
- Compute sup and inf of:  
  \[
  G_x(\theta, p) = \frac{\sum_{j=1}^{n+1} F(x; \theta_j) l_x(\theta_j) p_j}{\sum_{j=1}^{n+1} l_x(\theta_j) p_j}
  \]  
  under linear constraints on θ and p.  
- **Four-step algorithm**:  
  1. Candidate identification (independent of x).  
  2. Selection of the best candidates according to Gx.  
  3. Constrained optimization to find local maxima.  
  4. Approximation of sup and inf by interpolation over a discretized interval.  
- **Optimization** performed using `scipy.optimize.minimize` with `'trust-constr'` method.

---

## **4. Applications**
### **Application 1: Hydrology**
- **Data**: Annual daily maximum flows of the Meuse River (Belgium).  
- **Return levels**: 4 years, 0.1, and 0.001.  
- **Results**: Posterior quantile bounds computed using the proposed algorithm.

### **Application 2: Meteorology**
- **Data**: Annual daily maximum rainfall (Penta-di-Casinca, Corsica).  
- **Return levels**: 50 and 100 years.  
- **Results**: Posterior quantile bounds incorporating expert prior information.

---

## **5. References**
- Bruno Betrò, Fabrizio Ruggeri, and Marek Meczarski. *Robust Bayesian analysis under generalized moments conditions*. Journal of Statistical Planning and Inference, 41(3), 257–266, 1994. [DOI](https://doi.org/10.1016/0378-3758(94)90022-1)  

---

## **Repository Contents**
- **`theoretical_results.pdf`** – Detailed theoretical results.  
- **`numerical_methods.ipynb`** – Jupyter notebook with numerical simulations.  
- **`hydrology_data.csv`** – Hydrology application dataset.  
- **`meteorology_data.csv`** – Meteorology application dataset.  
- **`bayesian_utils.py`** – Utility functions for algorithm implementation.
