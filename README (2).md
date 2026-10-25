# 🔧 RepairWise — ML-Based Repair vs. Replace Decision Support

RepairWise is a machine learning-powered web application that helps users assess whether an electronic device may be a candidate for **repair, replacement, or professional inspection** based on its condition, estimated costs, and other relevant factors.

The project uses a **Decision Tree Classifier** trained on a 5,000-row synthetic dataset and provides an interactive web interface for entering device details and viewing predictions.

## ✨ Features

- **Device Assessment:** Enter device type, fault type, age, severity, and estimated costs.
- **Machine Learning Predictions:** Generate a recommendation category using a Decision Tree Classifier.
- **Data Preprocessing:** Handle missing values, encode categorical features, and scale numerical features.
- **Model Evaluation:** Evaluate performance on a held-out test set using accuracy, a classification report, and a confusion matrix.
- **Interactive Web Interface:** Access the model through a responsive website.
- **API Endpoints:** Provide health checks, model metrics, and device assessment predictions.
- **Vercel Deployment:** Deploy the web application online.

## 🧠 Machine Learning Workflow

1. Load the 5,000-row dataset.
2. Inspect and prepare the data.
3. Separate input features from the target variable.
4. Split the data into 75% training and 25% testing sets.
5. Apply preprocessing using a Scikit-learn pipeline.
6. Train a Decision Tree Classifier.
7. Evaluate the model using held-out test data.
8. Use the trained model to assess new device details.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data handling and numerical operations |
| Scikit-learn | Preprocessing, model training, and evaluation |
| FastAPI | Python API and application backend |
| HTML, CSS & JavaScript | Interactive frontend |
| Vercel | Deployment and hosting |

## 📁 Project Structure

```text
RepairWise/
├── app.py
├── repairwise_dataset.csv
├── requirements.txt
├── vercel.json
└── README.md
```

## 📊 Dataset

The project uses `repairwise_dataset.csv`, containing **5,000 synthetic records** with device and fault information, estimated repair and replacement costs, and recommendation labels.

The features include device type, fault type, device age, repair cost, replacement cost, fault severity, previous repairs, and parts availability.

**Important:** The dataset and its labels are synthetic and rule-generated. Model metrics measure agreement with these generated labels, not verified real-world repair accuracy.

## 🚀 Run Locally

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anushkamate/repairwise-vercel.git
   cd repairwise-vercel
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

4. Open the application:

   `http://127.0.0.1:8000`

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web application |
| `/api/health` | GET | Check application status |
| `/api/metrics` | GET | Retrieve model evaluation metrics |
| `/api/assess` | POST | Assess a device using the model |

The `/api/assess` endpoint accepts device information as JSON and returns a prediction and model scores.

## ☁️ Deployment on Vercel

1. Push all project files to a GitHub repository.
2. Sign in to [Vercel](https://vercel.com/).
3. Import the GitHub repository as a new project.
4. Keep the project root directory set to the repository root.
5. Deploy the project and review the build logs.
6. Open the generated deployment URL and test the application.

## 🔮 Future Improvements

- Collect real repair records verified by qualified technicians.
- Compare Decision Tree performance with other classification algorithms.
- Improve model evaluation using representative and independent real-world test data.
- Add user accounts and repair history.
- Include warranty, spare-parts pricing, and environmental impact considerations.
- Improve explainability by showing the factors influencing each recommendation.

## ⚠️ Disclaimer

RepairWise is an educational machine learning prototype. It is **not a substitute for professional diagnosis or repair advice**. Its predictions have not been validated on real-world repair outcomes. Always consult a qualified technician, particularly for electrical, battery, or other safety-related faults.

## 👩‍💻 Author

**Anushka Mate**

GitHub: [anushkamate](https://github.com/anushkamate)
