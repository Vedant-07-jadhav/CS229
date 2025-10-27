<div align="left" style="position: relative;">
<h1>CS229: Machine Learning</h1>

<p align="left">Built with the tools and technologies:</p>
<p align="left">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter">
</p>
</div>
<br clear="right">

## 📋 Quick Links

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
  - [Project Index](#-project-index)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
  - [Usage](#-usage)
- [Problem Sets](#-problem-sets)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Overview

This repository contains comprehensive documentation, code implementations, and problem set solutions for Stanford's CS229: Machine Learning course (2018). The project includes implementations of fundamental machine learning algorithms from scratch, along with practical applications and analysis.

**Course Topics Covered:**
- Supervised Learning (Linear Regression, Logistic Regression, GDA)
- Support Vector Machines
- Neural Networks
- Unsupervised Learning (K-Means, GMM, ICA)
- Reinforcement Learning (Cart-Pole problem)
- Naive Bayes and Spam Classification

---

## ✨ Features

- **Complete Problem Set Solutions**: All four problem sets with detailed implementations
- **From-Scratch Implementations**: Custom implementations of ML algorithms without high-level libraries
- **Interactive Notebooks**: Jupyter notebooks for exploration and visualization
- **Real-World Applications**: Spam classification, audio source separation, and more
- **Course Materials**: Class notes, extra notes, and syllabus included

---

## 📁 Project Structure

```sh
└── CS229/
    ├── Class_Notes/              # Lecture notes and materials
    ├── Extra_notes/              # Supplementary learning resources
    ├── NoteBooks/                # Interactive Jupyter notebooks
    │   ├── lect_2/              # Linear regression implementations
    │   ├── lect_3/              # Logistic regression
    │   ├── From_scratch.ipynb   # Algorithm implementations from scratch
    │   └── K_means_clustering.ipynb
    ├── problem_sets/             # All problem set solutions
    │   ├── PS1/                 # Linear models, GDA, Poisson regression
    │   ├── PS2/                 # SVM, Perceptron, Spam classification
    │   ├── PS3/                 # Neural Networks, GMM
    │   └── PS4/                 # ICA, Reinforcement Learning
    ├── MTfinalTBA_aut_2018.pdf  # Midterm exam
    ├── ps1.pdf                   # Problem set 1 questions
    ├── cs229_2018_syllabus.html # Course syllabus
    ├── environment.yml           # Conda environment specification
    └── README.md
```

### 📑 Project Index

<details open>
<summary><b><code>CS229/</code></b></summary>

<details>
<summary><b>problem_sets</b></summary>
<blockquote>

<details>
<summary><b>PS1 - Supervised Learning Basics</b></summary>
<blockquote>

**Topics**: Linear Regression, Logistic Regression, Gaussian Discriminant Analysis, Locally Weighted Regression

**Key Files**:
- `p01b_logreg.py` - Logistic regression implementation
- `p01e_gda.py` - Gaussian Discriminant Analysis
- `p02cde_posonly.py` - Learning from positive and unlabeled examples
- `p03d_poisson.py` - Poisson regression for count data
- `p05b_lwr.py` - Locally weighted regression
- `p05c_tau.py` - Bandwidth selection for LWR
- `linear_model.py` - Base linear model class
- `util.py` - Utility functions for plotting and data handling

</blockquote>
</details>

<details>
<summary><b>PS2 - Classification & SVMs</b></summary>
<blockquote>

**Topics**: Support Vector Machines, Kernel Methods, Spam Classification, Perceptron

**Key Files**:
- `p01_lr.py` - Logistic regression with Newton's method
- `svm.py` - Support Vector Machine implementation
- `p05_percept.py` - Perceptron algorithm with dot and RBF kernels
- `p06_spam.py` - Naive Bayes spam classifier
- `p01_solutions.ipynb` - Detailed problem solutions

**Outputs**:
- Trained model predictions
- Feature importance analysis
- Optimal hyperparameters (kernel radius, etc.)

</blockquote>
</details>

<details>
<summary><b>PS3 - Unsupervised Learning</b></summary>
<blockquote>

**Topics**: Neural Networks, Gaussian Mixture Models, K-Means

**Key Files**:
- `p01_nn.py` - Neural network implementation and training
- `p03_gmm.py` - Gaussian Mixture Model with EM algorithm

</blockquote>
</details>

<details>
<summary><b>PS4 - Advanced Topics</b></summary>
<blockquote>

**Topics**: Independent Component Analysis, Reinforcement Learning

**Key Files**:
- `p04_ica.py` - ICA for blind source separation (audio unmixing)
- `p06_cartpole.py` - Reinforcement learning for cart-pole balancing
- `env.py` - Custom environment for RL experiments
- `make_zip.py` - Submission packaging script

**Outputs**:
- Separated audio sources (split_*.wav)
- Mixed audio samples (mixed_*.wav)
- Learned unmixing matrix (W.txt)

</blockquote>
</details>

</blockquote>
</details>

<details>
<summary><b>NoteBooks</b></summary>
<blockquote>

**Interactive Learning Materials**:
- `From_scratch.ipynb` - ML algorithms implemented from first principles
- `K_means_clustering.ipynb` - Clustering analysis and visualization

**Lecture-Specific Notebooks**:
- `lect_2/linear_regression.ipynb` - Simple linear regression
- `lect_2/Multiple_linear_regression.ipynb` - Multivariate regression
- `lect_2/locallyweighted_regression.ipynb` - Non-parametric regression
- `lect_3/logistic_regression.ipynb` - Binary classification

</blockquote>
</details>

</details>

---

## 🚀 Getting Started

### Prerequisites

**System Requirements**:
- Python 3.7 or higher
- Conda package manager
- 4GB RAM minimum (8GB recommended for larger datasets)
- Git for version control

**Required Knowledge**:
- Linear algebra fundamentals
- Calculus and optimization basics
- Python programming
- Basic probability and statistics

### Installation

**Option 1: Using Conda (Recommended)**

1. Clone the repository:
```sh
git clone https://github.com/Vedant-07-jadhav/CS229.git
cd CS229
```

2. Create and activate the conda environment:
```sh
conda env create -f environment.yml
conda activate cs229
```

**Option 2: Using pip**

1. Clone the repository:
```sh
git clone https://github.com/Vedant-07-jadhav/CS229.git
cd CS229
```

2. Install dependencies:
```sh
pip install numpy scipy matplotlib jupyter pandas scikit-learn
```

### Usage

**Running Problem Sets**

Navigate to a specific problem set and run the Python scripts:

```sh
cd problem_sets/PS1/src
python p01b_logreg.py  # Run logistic regression
```

**Using Jupyter Notebooks**

Launch Jupyter and explore the interactive notebooks:

```sh
jupyter notebook
# Navigate to NoteBooks/ folder in the browser
```

**Example: Training a Logistic Regression Model**

```python
from problem_sets.PS1.src.p01b_logreg import LogisticRegression
from problem_sets.PS1.src.util import load_dataset

# Load data
x_train, y_train = load_dataset('data/ds1_train.csv', add_intercept=True)

# Train model
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Make predictions
predictions = clf.predict(x_train)
```

---

## 📚 Problem Sets

### PS1: Supervised Learning Foundations
- **Topics**: Linear models, GDA, Poisson regression, locally weighted regression
- **Highlights**: Implementing classification from positive-only data, comparing generative vs. discriminative models

### PS2: SVMs and Spam Classification
- **Topics**: Support Vector Machines, kernel methods, Naive Bayes
- **Highlights**: Building a spam filter, comparing linear and RBF kernels

### PS3: Neural Networks and Clustering
- **Topics**: Deep learning basics, Gaussian Mixture Models
- **Highlights**: Backpropagation from scratch, EM algorithm implementation

### PS4: Advanced ML Applications
- **Topics**: Independent Component Analysis, Reinforcement Learning
- **Highlights**: Cocktail party problem (audio source separation), training RL agents

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new implementations, or improved documentation.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Create your own fork of the project
   ```sh
   # Click 'Fork' on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/CS229.git
   ```

2. **Create a Feature Branch**: Work on a descriptive branch
   ```sh
   git checkout -b feature/improved-svm-implementation
   ```

3. **Make Your Changes**: 
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test Your Changes**:
   ```sh
   python -m pytest tests/  # If tests exist
   ```

5. **Commit with Clear Messages**:
   ```sh
   git commit -m "Add RBF kernel optimization to SVM"
   ```

6. **Push and Create PR**:
   ```sh
   git push origin feature/improved-svm-implementation
   ```
   Then open a Pull Request on GitHub with a detailed description

</details>

### Ways to Contribute

- 🐛 Report bugs and issues
- 📝 Improve documentation and code comments
- ✨ Add new algorithm implementations
- 🧪 Create test cases
- 💡 Suggest optimizations and improvements

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/Vedant-07-jadhav/CS229/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Vedant-07-jadhav/CS229">
   </a>
</p>
</details>

---

## 📄 License

This project is for educational purposes. Course materials are property of Stanford University. Code implementations are available under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Stanford CS229**: Professor Andrew Ng and the teaching staff
- **Course Materials**: Stanford University (2018)
- **Community**: All contributors and students who have improved this repository
- **Libraries**: NumPy, SciPy, Matplotlib communities

---

## 📬 Contact

For questions or discussions about this repository:
- **Issues**: [Report bugs or request features](https://github.com/Vedant-07-jadhav/CS229/issues)
- **Discussions**: [Join community discussions](https://github.com/Vedant-07-jadhav/CS229/discussions)

---

<div align="center">

**[⬆ Back to Top](#-quick-links)**

</div>