# Rain Prediction Neural Network for ESP32 (1-Hour Ahead Edge Forecasting)

A lightweight, standalone Deep Learning model built with Keras / TensorFlow and optimized for **direct Edge AI execution on ESP32 microcontrollers using MicroPython**. 

Unlike full cloud-dependent weather models, this localized version predicts rainfall 1 hour in advance using **only physical parameters directly measurable by standard local sensors** (BME280 temperature, humidity, and atmospheric pressure) alongside time variables. It eliminates all dependencies on external cloud APIs (such as cloud cover, wind speed, or current rain radar data).

---

## 1. Project Purpose & Scope

* **Target Device**: ESP32 microcontroller running MicroPython.
* **Task**: Binary Classification (0 = No Rain, 1 = Rain in 1 hour).
* **Operational Mode**: 100% Offline / Standalone Edge Inference (no Wi-Fi/API required once deployed).
* **Key Difference from Previous Model**: Removed unmeasurable local features (`cloud_cover`, `wind_speed_10m`, `is_raining_now`). Replaced them with local temporal dynamics (`pressure_change_3h`) to infer approaching weather fronts purely from local barometric trends.

---

## 2. Model Architecture & Features

### Input Features (10 local parameters):
1. **Time Encoding**: `hour_sin`, `hour_cos`, `month_sin`, `month_cos` (computed algorithmically from RTC / NTP).
2. **Direct Sensor Values**: `temperature_2m (°C)`, `relative_humidity_2m (%)`, `surface_pressure (hPa)`.
3. **Local Dynamics (Deltas)**:
   * `pressure_change`: 1-hour barometric delta ($P_t - P_{t-1}$).
   * `humidity_change`: 1-hour relative humidity delta ($H_t - H_{t-1}$).
   * `pressure_change_3h`: 3-hour long-term barometric trend ($P_t - P_{t-3}$).

### Neural Network Structure:
* **Input Layer**: 10 scaled features (`StandardScaler`).
* **Hidden Layer**: `Dense(32, activation='relu')`.
* **Output Layer**: `Dense(1, activation='sigmoid')`.
* **Optimizer**: Adam (`learning_rate=0.0005`).
* **Loss Function**: `BinaryCrossentropy()`.
* **Class Weighting**: Balanced (`compute_class_weight`).

---

## 3. Local Model Development & Optimization Iterations

| Iteration | Description / Architecture Changes | Accuracy | AUC | Loss | Precision | Recall | Deployment Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **v1. Local Baseline** | Removed API features (9 local features), LR=0.0005, 50 epochs | 0.8143 | 0.8553 | 0.4668 | 0.4058 | 0.7041 | Solid baseline for standalone offline operation. |
| **v2. Added 3h Pressure Delta** | Added `pressure_change_3h` (10 features) | **0.8216** | **0.8704** | **0.4429** | **0.4223** | **0.7328** | **Best balanced model.** 3h pressure trend significantly improved front detection. |
| **v3. Extra Hidden Layer** | Added `Dense(16, relu)` (2 hidden layers) | 0.8109 | 0.8748 | 0.4325 | 0.3999 | 0.7272 | *Rejected.* Marginal AUC gain but lower Precision and doubled memory/compute overhead on ESP32. |

---

## 4. Final Selected Configuration

The **v2 model architecture** was selected for final deployment:
* **Features**: 10 local inputs (including 1h and 3h pressure trends).
* **Single Hidden Layer**: `Dense(32, ReLU)` to minimize RAM/flash footprint on ESP32.
* **Performance**: AUC ~0.87, Recall ~73.3%, Precision ~42.2%.
* **Stability**: Smooth convergence with `learning_rate=0.0005` over 50 epochs.

---

## 5. MicroPython Deployment Strategy

To run inference on ESP32 without C++ compilation or heavy TensorFlow Lite runtimes:
1. **Weight & Scaler Export**: Weights ($W_1, b_1, W_2, b_2$) and normalization statistics (`mean`, `scale`) are saved into a lightweight `model_data_esp.json` file.
2. **Pure Python Forward Pass**: A custom MicroPython function standardizes raw sensor readings and computes matrix multiplications in under 1 ms:
   $$y = \text{sigmoid}(W_2 \cdot \text{ReLU}(W_1 \cdot x_{scaled} + b_1) + b_2)$$
3. **Memory Footprint**: Uses < 5 KB RAM and < 10 KB flash memory.

---

## 6. Project Roadmap & Future Hardware Integration

* [x] Feature selection for standalone local sensors.
* [x] Model training and hyperparameter optimization.
* [x] Weight export to JSON for MicroPython.
* [ ] Hardware assembly (ESP32 + BME280).
* [ ] MicroPython driver integration & sensor reading loop.
* [ ] Field testing & live prediction validation.