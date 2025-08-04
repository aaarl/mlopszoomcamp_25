from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import pickle

class TrainCancerModelFlow(FlowSpec):
    
    n_estimators = Parameter("n_estimators", default=100)

    @step
    def start(self):
        print("🚀 Starting Metaflow pipeline for breast cancer classification...")
        self.next(self.load_data)

    @step
    def load_data(self):
        print("📥 Loading dataset...")
        df = pd.read_csv("data/breast_cancer.csv")
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.next(self.train_model)

    @step
    def train_model(self):
        print(f"🧠 Training RandomForest with {self.n_estimators} trees...")
        model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("📊 Evaluating model...")
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        self.accuracy = acc
        print(f"✅ Accuracy: {acc:.4f}")
        self.next(self.log_mlflow)

    @step
    def log_mlflow(self):
        print("📦 Logging model and metrics to MLflow...")
        mlflow.set_experiment("breast_cancer_detection")
        with mlflow.start_run() as run:
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(self.model, "model")

            # Optional: Save model locally
            os.makedirs("models", exist_ok=True)
            with open("models/model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            print(f"📁 Model saved locally and logged to MLflow run ID: {run.info.run_id}")

        self.next(self.end)

    @step
    def end(self):
        print("🏁 Pipeline completed successfully.")

if __name__ == "__main__":
    TrainCancerModelFlow()
