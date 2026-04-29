FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . .
COPY ./mlflow_runs/mlruns/models/. ./mlflow_runs/mlruns/models/
COPY ./mlflow_runs/mlruns/896236271089450697/7edda145087b428f97b3f753e1301f0b/. ./mlflow_runs/mlruns/896236271089450697/7edda145087b428f97b3f753e1301f0b/
COPY ./mlflow_runs/mlruns/896236271089450697/models/m-d2b1b983ece842189a76e5eb1fce4715/. ./mlflow_runs/mlruns/896236271089450697/models/m-d2b1b983ece842189a76e5eb1fce4715/
COPY ./mlflow_runs/mlruns/896236271089450697/meta.yaml ./mlflow_runs/mlruns/896236271089450697/
COPY ./mlflow_runs/mlruns/0/. ./mlflow_runs/mlruns/0/
COPY ./mlflow_runs/mlruns/.trash/. ./mlflow_runs/mlruns/.trash/
COPY ./train_savings/model_5/. ./train_savings/model_5/
EXPOSE 8000
CMD ["uvicorn", "modules.API.app:app", "--host", "0.0.0.0", "--port", "8000"]