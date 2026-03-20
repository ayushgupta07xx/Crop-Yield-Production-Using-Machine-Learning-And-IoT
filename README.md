# 🌾 Crop-Yield-Prediction-Using-Machine-Learning-And-IoT

An intelligent system integrating **Machine Learning (ML)** and the **Internet of Things (IoT)** to predict crop yields using real-time soil, weather, and crop data. It features a **Flask web app** and **IoT sensors** for data collection, providing farmers with accurate forecasts, insights, and automation for smart, sustainable farming.

---

## 🚀 Overview
This project combines predictive analytics and IoT-based monitoring to forecast crop yields and assist in agricultural decision-making. It leverages machine learning models trained on historical agricultural datasets and integrates real-time sensor data to provide precise, adaptive, and actionable predictions.

---

## 🔑 Features
- 🌱 **Accurate Yield Prediction** using the CatBoost regression algorithm  
- ☁️ **IoT Data Integration** with NPK, moisture, temperature, and rainfall sensors  
- 🧠 **Machine Learning + Real-time Monitoring** for adaptive forecasting  
- 💻 **Flask Web App** with user authentication and a data visualization dashboard  
- ⚙️ **Automation Support** for irrigation and fertilizer alerts  
- 🗄️ **MongoDB Database** for secure and scalable data storage  

---

## 🧠 Tech Stack
| Category | Technology |
|-----------|-------------|
| Programming Language | Python 3.x |
| ML Framework | CatBoost, Scikit-learn, Pandas, NumPy |
| Web Framework | Flask, HTML, CSS, JavaScript |
| Database | MongoDB |
| Hardware | NodeMCU (ESP8266/ESP32), NPK, DHT11, BMP280, Soil Moisture Sensor |
| Hosting | Local / Cloud (Heroku / PythonAnywhere) |

---

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/your-username/Crop-Yield-Prediction-Using-Machine-Learning-And-IoT.git

# Navigate to the project directory
cd Crop-Yield-Prediction-Using-Machine-Learning-And-IoT

# Install dependencies
pip install flask flask-sqlalchemy flask-login flask-wtf flask-mail catboost scikit-learn matplotlib seaborn
pip install email_validator
python 3.10

# Run the Flask web app
python main.py

```

## 🌐 Usage
1. Start the Flask server.  
2. Open the web app in your browser.  
3. Register or log in to your account.  
4. Enter crop type, location, and season to get yield predictions.  
5. View real-time IoT sensor data and insights.  
6. Receive alerts or automated irrigation control based on predictions.  

---

## 📊 System Workflow
1. IoT sensors collect environmental and soil data.  
2. The Flask server receives and stores it in MongoDB.  
3. The ML model (CatBoost) processes the data and predicts yield.  
4. Results and recommendations are displayed on the web dashboard.  

---

## 👨‍💻 Contributors
- **Antara Chakravorthy (22BCB0020)**  
- **Umar Khan (22BCE0501)**  
- **Harshini Sunkara (22BCE2373)**
- **Soumya ajmera (22BCT0005)**

---

## 📜 License
This project was developed as part of **VIT B.Tech Project BCSE497J (2025)**.  
© 2025 Crop Yield Prediction Team. All rights reserved.

