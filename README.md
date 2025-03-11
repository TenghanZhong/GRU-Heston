# GRU-Heston
A full implementation of hybrid Heston-GRU model for ETF option pricing.

**README for Hybrid ETF Option Pricing Model Repository**  

---

# 📌 **Hybrid ETF Option Pricing Model: Heston + GRU**  

This repository provides a **comprehensive end-to-end pipeline** for **ETF option pricing**, integrating the **Heston stochastic volatility model** with a **Gated Recurrent Unit (GRU) network** to improve pricing accuracy. By leveraging **high-frequency time series data**, the model dynamically corrects traditional pricing errors, enhancing market efficiency and prediction reliability.  

---

## 🔹 **Project Overview**  

Traditional option pricing models, such as **Black-Scholes** and **Heston**, often struggle to fully capture the complex, non-linear nature of **high-frequency ETF option prices**. This project proposes a **hybrid correction framework**, where a **GRU model** learns and adjusts for pricing deviations in real-time.  

### 🔹 **Key Contributions:**  
✔️ **Data cleaning and preprocessing of high-frequency option & ETF spot price data**  
✔️ **Using high frequency data for parameter calibration of Heston model via WMSE minimization**  
✔️ **Error correction using GRU to refine pricing estimates dynamically**  
✔️ **Integration of deep learning to enhance traditional financial model**  

---

## 🚀 **Workflow**  

### **1️⃣ Data Cleaning & Preprocessing**  
- Collects **high-frequency ETF option data** via **Tushare API**  
- Merges **ETF spot prices** and computes **time-to-maturity (TTM)**  
- Estimates **risk-free rates (Rf)** using **linear interpolation on SHIBOR**  

### **2️⃣ Heston Model Calibration**  
- Uses **Trust-Region Reflective (TRR) algorithm** to optimize the **five Heston parameters**  
- Minimizes **Weighted Mean Squared Error (WMSE)** between **Heston prices and market prices**  

### **3️⃣ Bias Extraction & Normalization**  
- Computes **pricing errors** between Heston estimates and real market prices  
- Applies **min-max normalization** to handle the **non-Gaussian distribution** of errors  

### **4️⃣ GRU Model Training**  
- **Hyperparameter tuning** via **grid search & 10-fold cross-validation**  
- Trains GRU on the **last 60 minutes of error sequences** to capture temporal dependencies  

### **5️⃣ Pricing Correction & Refinement**  
- Predicts **next-minute error (t+1) based on the past hour’s data**  
- Corrects the **Heston price estimate** by adding **GRU-predicted deviations**  
- Produces a **refined ETF option price at minute-level granularity**  

---

## 📈 **Results & Performance**  

✔️ The **hybrid Heston-GRU model** significantly **reduces pricing errors**, achieving:  
- **More accurate** ETF option pricing at a high-frequency level  
- **Enhanced generalization** compared to traditional pricing models  
- **Greater robustness** in handling market volatility and nonlinear dependencies  

---

## 🔧 **Installation & Usage**  

1. Clone this repository:  
   ```bash
   git clone https://github.com/TenghanZhong/tenghanz.github.io.git
   cd tenghanz.github.io
   ```  
2. Install dependencies:  
   ```bash
   pip install numpy pandas tushare matplotlib QuantLib torch
   ```  
3. Run **data preprocessing & Heston calibration**:  
   ```bash
   python data_cleaning.py
   python heston_calibration.py
   ```  
4. Train **GRU model & refine option prices**:  
   ```bash
   python train_gru.py
   python pricing_correction.py
   ```  

---

## 📜 **Repository Structure**  

```plaintext
📂 tenghanz.github.io/
│── 📜 README.md               # Project documentation  
│── 📜 requirements.txt        # Dependencies  
│── 📂 data/                   # Processed high-frequency ETF option data  
│── 📂 scripts/                # Core modeling scripts  
│   │── data_cleaning.py       # Tushare API & data preprocessing  
│   │── heston_calibration.py  # Heston model calibration  
│   │── train_gru.py           # GRU model training  
│   │── pricing_correction.py  # GRU-based pricing adjustment  
│── 📂 models/                 # Saved model weights  
│── 📂 results/                # Evaluation & visualization outputs  
```  

---

## 🏆 **Key Features**  

✔️ **Full Pipeline**: Data Cleaning → Heston Calibration → GRU Correction  
✔️ **High-Frequency Time Series Handling**  
✔️ **Bridges Financial Model Transparency with Deep Learning Adaptability**  
✔️ **Python-based Implementation with Tushare, NumPy, Tensorflow & QuantLib**  

---

## 📬 **Contact & Citation**  

If you find this work helpful, feel free to reach out via **tenghanz@usc.edu \ k3802286782@gmail.com** or contribute to the repository!  
