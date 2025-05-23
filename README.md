# 🔄 Event-Based Synchronization for Network Digital Twins

This project provides a comparative evaluation of multiple synchronization techniques for **event-based time-series data**, aimed at supporting the development and accuracy of **Network Digital Twins** (NDTs). 

By simulating and assessing the performance of each method, we aim to determine the most effective approach for minimizing error while optimizing synchronization efficiency.

## 🎯 Project Objective

To evaluate and compare various techniques for event-based synchronization, including:

- **Adaptive Synchronization**
- **Model Predictive Control (MPC)**
- **Kalman Filter-based Synchronization**
- **Reinforcement Learning-based Synchronization (PPO, SAC)**

Each method is tested on time-series input data (e.g., packets per second) to determine its effectiveness in:

- Maintaining low prediction error (MAE, RMSE)
- Reducing unnecessary synchronization events
- Supporting reliable updates in Network Digital Twins

## 🧪 Evaluation Focus

- 📉 **Error Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
- 🔁 **Synchronization Cost**: Number of synchronization events triggered
- ⚖️ **Trade-Off Analysis**: Balancing accuracy and update frequency

## 💡 Why This Matters

In the context of 5G/6G networks and emerging **Digital Twin** technologies, keeping virtual replicas accurately synchronized with real-world network conditions is essential. However, frequent updates can be resource-intensive.

This project explores **which synchronization strategy best supports this balance**, guiding future integration in NDT systems.


## 📈 Example Outcome (Preview)

After running the simulations, you will obtain:

- Performance comparison plots
- Synchronization count charts
- Summary tables of evaluation metrics

---

📌 *project is under active development. More features, automation, and documentation will follow shortly.*

