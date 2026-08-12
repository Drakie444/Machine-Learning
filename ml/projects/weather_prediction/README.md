# Rain Prediction Neural Network (1-Hour Ahead Forecasting)

A lightweight Deep Learning model built with Keras / TensorFlow to predict rainfall occurrence 1 hour in advance using tabular meteorological data. The model is optimized for high accuracy, balanced Precision-Recall trade-offs..

---

## 1. Project Overview & Architecture

* **Task**: Binary Classification (0 = No Rain, 1 = Rain in 1 hour).
* **Dataset**: Open-Meteo hourly weather observations (temperature, relative humidity, surface pressure, wind speed, cloud cover, timestamps).
* **Model Architecture**:
  * **Input Layer**: 12 normalized features.
  * **Hidden Layer**: `Dense(32, activation='relu')` for capturing non-linear meteorological patterns.
  * **Output Layer**: `Dense(1, activation='sigmoid')` producing probability of rain.
  * **Optimizer**: Adam (`learning_rate=0.001`).
  * **Loss Function**: `BinaryCrossentropy()`.

---

## 2. Feature Engineering & Preprocessing

1. **Cyclical Time Encoding**: Converted `hour` (0–23) and `month` (1–12) into `sin` and `cos` trigonometric components to capture daily and seasonal continuity.
2. **Current Rain State**: Added `is_raining_now` flag (`rain (mm) > 0`).
3. **Meteorological Deltas**: Calculated 1-hour rate of change for pressure and humidity (`diff(1)`):
   * `pressure_change`: $P_t - P_{t-1}$ (drop in pressure signals approaching fronts).
   * `humidity_change`: $H_t - H_{t-1}$ (rapid humidity increase).
4. **Target Shift**: Shifted target label by $-1$ hour (`shift(-1)`) to forecast rainfall occurrence 1 hour ahead.
5. **Feature Scaling**: Applied `StandardScaler` on all 12 input features.

---

## 3. Iterative Development & Performance Evolution

### Iteration 1: Baseline Model (Single Neuron)
* **Setup**: `Dense(1, sigmoid)`, unweighted, raw target.
* **Result**: Model collapsed into majority class (predicting "no rain" everywhere).
* **Stats**: Accuracy: 84.85% | AUC: 0.6960 | Loss: 0.3812 | Precision: 0.4426 | Recall:  0.1226
* **Issue**: Zero predictive capability due to class imbalance (~15% rain prevalence).



### Iteration 2: Added Hidden Layer (`Dense(32, relu)`)
* **Setup**: Added hidden layer to capture non-linear feature interactions, LR = 0.01.
* **Result**: Significant boost in AUC and Precision, but Recall remained low.
* **Stats**: Accuracy: 84.28% | AUC: 0.8152 | Loss: 0.1782 | Precision: 0.5618 | Recall: 0.1236
* **Issue**: Model remained overly conservative due to unweighted loss function.

### Iteration 3: Class Weights Attempt & Bug Encounter
* **Setup**: Introduced `compute_class_weight('balanced')`, LR = 0.01.
* **Result**: Training failed completely (Loss: -130798 / -5.40, AUC dropped to ~0.50–0.69).
* **Root Causes**:
  1. Target was continuous (`rain (mm)`) instead of binary (`0` or `1`), causing invalid negative binary cross-entropy.
  2. Input features were unscaled (`x_train` used instead of `x_train_scaled`).
  3. LR was too high (0.01) when multiplied by class weights.

### Iteration 4: Bug Fixes & Balanced Class Training
* **Setup**: Converted target to binary `(rain > 0)`, applied `StandardScaler`, set threshold = 0.6.
* **Result**: Major breakthrough — Recall surged to ~76–85%, AUC exceeded 0.91.
* **Stats**: Accuracy: 85.73% | AUC: 0.9114 | Loss: 0.3711 | Precision: 0.4959 | Recall: 0.7615
* **Issue**: Validation loss curve exhibited sharp spikes due to high learning rate (`0.01`).

### Iteration 5 (Final): 1-Hour Ahead Target + Delta Features + LR Tuning
* **Setup**: Shifted target (`shift(-1)`), added `pressure_change` & `humidity_change`, reduced LR to `0.001`, threshold = 0.6.
* **Result**: Smooth convergence, zero oscillations, high stability.
* **Stats**: Accuracy: ~81–86% | **Val AUC: 0.925+** | Loss: ~0.316 (smooth) | Precision: ~0.45–0.50 | Recall: ~0.80–0.85

---

## 5. Summary & Key Takeaways

1. **Non-linearity matters**: Adding a single 32-neuron ReLU layer increased AUC from 0.63 to 0.81.
2. **Proper data prep is crucial**: Binary cross-entropy requires strictly binary targets ($0/1$) and standardized features.
3. **Addressing class imbalance**: Balanced class weighting boosted Recall from ~12% to over 80%.
4. **Learning Rate adjustment**: Reducing LR from 0.01 to 0.001 eliminated validation loss spikes while preserving top AUC performance.
5. **Feature engineering beats model complexity**: Adding 1-hour deltas (`pressure_change`, `humidity_change`) boosted Val AUC to a peak of 0.925+.

