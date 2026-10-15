# 🏀 NBA Win Prediction with Logistic Regression

Predicting NBA game outcomes (win/loss) from box-score statistics. The project starts with a one-feature baseline (points scored) and builds up to a multivariate model, raising **accuracy from 70% to 86%** and **ROC-AUC from 0.76 to 0.93**.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 🎯 Goal

Estimate the probability that a team wins a game, given its box-score stats:

$$\hat{y} = P(\text{Win} = 1 \mid X)$$

**Why logistic regression?** It outputs win *probabilities*, not just labels, and gives a simple decision rule: predict a win when $\hat{y} \ge 0.5$. Its coefficients are also easy to interpret, which makes it a strong, transparent baseline.

---

## 📊 Results

Evaluated on a **stratified 80/20 train/test split**.

| Model | Features | Accuracy | ROC-AUC | Log Loss |
|---|---|:---:|:---:|:---:|
| Baseline | `PTS` only | 0.6992 | 0.7550 | 0.5952 |
| **Full model** | Box score (`FGM` → `PTS`), excluding `PLUS_MINUS` | **0.8577** | **0.9298** | — |
| **Improvement** | | **+15.9 pts** | **+0.175** | |

<details>
<summary><b>Baseline confusion matrix</b></summary>

| | Predicted Loss | Predicted Win |
|---|:---:|:---:|
| **Actual Loss** | 177 | 69 |
| **Actual Win** | 79 | 167 |

</details>

---

## 🧠 Modeling Choices

- **Baseline first.** A points-only model gives a reference point, so every added feature has to earn its place.
- **Intercept included.** Lets the decision boundary shift instead of being forced through the origin.
- **Loss functions compared.** Plotted the loss surfaces for **mean squared error** and **cross-entropy (log loss)** to show why cross-entropy is the right objective for classification: it's convex and gives well-behaved gradients.
- **Leakage control.** Excluded `PLUS_MINUS`. It encodes the point differential, which effectively *is* the outcome, so it would make the model look near-perfect while being useless for prediction.
- **Multicollinearity.** Box-score features like `PTS`, `FGM`, `FGA`, and `FG%` are strongly correlated, so some individual coefficients look counterintuitive. They are conditional effects and should be read with caution unless regularization or feature selection is applied.

---

## 🔭 Next Steps

- Add **L1/L2 regularization** to stabilize coefficients under multicollinearity
- Use **pre-game features** (rolling team averages, rest days, home/away) for true forecasting
- Compare against tree-based models (random forest, gradient boosting)

---

## 📬 Contact

**Suhas Makineni** · [GitHub](https://github.com/skmakineni3) · [skmakineni@gmail.com](mailto:skmakineni@gmail.com)
