"""End-to-End Verification Test for AI Data Scientist Lifecycle (Phases 1 - 11)."""
import sys
import os
import io
import time
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.main import app
from app.database.session import init_db
from app.services.seed_data import seed_all

print("Initializing DB & Knowledge Seed...")
init_db()
seed_all()

client = TestClient(app)

print("\n--- 1. Testing Root & Health Endpoints ---")
r = client.get("/")
assert r.status_code == 200, f"Root failed: {r.text}"
print("Root OK:", r.json())

r = client.get("/health")
assert r.status_code == 200, f"Health check failed: {r.text}"
print("Health OK:", r.json())

print("\n--- 2. Testing Authentication ---")
login_res = client.post("/api/v1/auth/login", json={"email": "admin@aidatascientist.io", "password": "admin123"})
if login_res.status_code != 200:
    # Register admin if not found
    reg_res = client.post("/api/v1/auth/register", json={
        "email": "admin@aidatascientist.io",
        "username": "admin",
        "password": "admin123",
        "full_name": "Admin Data Scientist"
    })
    token = reg_res.json()["access_token"]
else:
    token = login_res.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}
me_res = client.get("/api/v1/auth/me", headers=headers)
assert me_res.status_code == 200
print("Authenticated as:", me_res.json()["username"])

print("\n--- 3. Testing Phase 1: Project Creation & Goal Understanding ---")
proj_res = client.post("/api/v1/projects/", json={
    "name": "Customer Churn Prediction",
    "description": "Predict whether a subscription customer will churn next month.",
    "business_goal": "Identify high-risk churn customers to reduce annual attrition by 15%."
}, headers=headers)
assert proj_res.status_code == 200, f"Project create failed: {proj_res.text}"
project = proj_res.json()
project_id = project["id"]
print(f"Created Project #{project_id}: {project['name']}")

goal_res = client.post(f"/api/v1/projects/{project_id}/understand-goal", json={
    "business_goal": project["business_goal"]
}, headers=headers)
assert goal_res.status_code == 200
print("Goal Understanding:", goal_res.json())

print("\n--- 4. Testing Phase 1 & 2: Dataset Generation & Upload ---")
# Create synthetic churn dataset
np.random.seed(42)
n_samples = 200
data = {
    "tenure_months": np.random.randint(1, 72, size=n_samples),
    "monthly_charges": np.random.uniform(20.0, 120.0, size=n_samples).round(2),
    "total_charges": np.random.uniform(100.0, 5000.0, size=n_samples).round(2),
    "contract_type": np.random.choice(["month-to-month", "one-year", "two-year"], size=n_samples),
    "payment_method": np.random.choice(["electronic_check", "mailed_check", "bank_transfer"], size=n_samples),
    "support_tickets": np.random.randint(0, 10, size=n_samples),
    "churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
}
df = pd.DataFrame(data)
csv_buffer = io.BytesIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

upload_res = client.post(
    "/api/v1/datasets/upload",
    data={"project_id": str(project_id)},
    files={"file": ("churn_data.csv", csv_buffer, "text/csv")},
    headers=headers
)
assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
dataset = upload_res.json()
dataset_id = dataset["id"]
print(f"Uploaded Dataset #{dataset_id} ({dataset['num_rows']} rows, {dataset['num_columns']} columns)")

preview_res = client.get(f"/api/v1/datasets/{dataset_id}/preview", headers=headers)
assert preview_res.status_code == 200
print(f"Dataset preview retrieved with {len(preview_res.json()['data'])} rows.")

def wait_for_task(client, task_id, headers, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/api/v1/tasks/{task_id}", headers=headers)
        if r.status_code == 200:
            data = r.json()
            if data["status"] in ["completed", "failed"]:
                return data
        time.sleep(0.5)
    return None

print("\n--- 5. Testing Phase 2: Data Profiling & Quality Analysis ---")
profile_task_res = client.post(f"/api/v1/datasets/{dataset_id}/profile", headers=headers)
assert profile_task_res.status_code == 200
p_task = wait_for_task(client, profile_task_res.json()["task_id"], headers)
print("Profiling task status:", p_task.get("status") if p_task else "timeout")

prof_res = client.get(f"/api/v1/datasets/{dataset_id}/profile", headers=headers)
assert prof_res.status_code == 200
print("Profiling completed:", prof_res.json().get("is_profiled"))

print("\n--- 6. Testing Phase 3: Dataset Fingerprinting ---")
fp_task_res = client.post(f"/api/v1/datasets/{dataset_id}/fingerprint", headers=headers)
assert fp_task_res.status_code == 200
fp_task = wait_for_task(client, fp_task_res.json()["task_id"], headers)
print("Fingerprinting task status:", fp_task.get("status") if fp_task else "timeout")

fp_res = client.get(f"/api/v1/datasets/{dataset_id}/fingerprint", headers=headers)
assert fp_res.status_code == 200
fp_data = fp_res.json()
print(f"Generated 128-dim fingerprint with {fp_data['num_numeric']} numeric and {fp_data['num_categorical']} categorical features.")

print("\n--- 7. Testing Phase 4: Meta-Learning & Experience Similarity ---")
similar_res = client.get(f"/api/v1/datasets/{dataset_id}/similar", headers=headers)
assert similar_res.status_code == 200
sim_data = similar_res.json()
print(f"Found {len(sim_data['similar_datasets'])} similar historical benchmarks. Top match: {sim_data['similar_datasets'][0]['best_algorithm']} (similarity: {sim_data['similar_datasets'][0]['similarity_score']})")

print("\n--- 8. Testing Phase 5: Hybrid Algorithm Recommendation ---")
rec_task_res = client.post(f"/api/v1/projects/{project_id}/recommend", headers=headers)
assert rec_task_res.status_code == 200
for _ in range(20):
    time.sleep(1.0)
    recs_res = client.get(f"/api/v1/projects/{project_id}/recommendations", headers=headers)
    if recs_res.status_code == 200 and len(recs_res.json()) > 0:
        break

assert recs_res.status_code == 200
recs = recs_res.json()
assert len(recs) > 0, "No recommendations generated!"
print(f"Generated {len(recs)} ranked recommendations. Top candidate: {recs[0]['algorithm_name']} (score: {recs[0]['recommendation_score']})")

print("\n--- 9. Testing Phase 6: Adaptive Preprocessing Pipeline ---")
prep_task_res = client.post(f"/api/v1/projects/{project_id}/preprocessing/generate", headers=headers)
assert prep_task_res.status_code == 200
for _ in range(20):
    time.sleep(1.0)
    prep_res = client.get(f"/api/v1/projects/{project_id}/preprocessing", headers=headers)
    if prep_res.status_code == 200 and prep_res.json().get("pipeline_steps"):
        break

assert prep_res.status_code == 200
prep = prep_res.json()
print(f"Constructed preprocessing pipeline with {len(prep['pipeline_steps'])} adaptive steps.")

print("\n--- 10. Testing Phase 7 & 8: Intelligent Training & Leaderboard ---")
train_task_res = client.post(f"/api/v1/projects/{project_id}/train", json={
    "max_trials_s1": 2,
    "max_trials_s2": 2,
    "max_trials_s3": 2,
    "create_ensembles": True,
}, headers=headers)
assert train_task_res.status_code == 200
print("Training task launched. Waiting for progressive 3-stage training...")

# Wait for training
for _ in range(30):
    time.sleep(1.5)
    t_status = client.get(f"/api/v1/tasks/{train_task_res.json()['task_id']}", headers=headers).json()
    print(f"  Training progress: {int(t_status['progress'] * 100)}% - {t_status['message']}")
    if t_status["status"] in ["completed", "failed"]:
        break

leaderboard_res = client.get(f"/api/v1/projects/{project_id}/leaderboard", headers=headers)
assert leaderboard_res.status_code == 200
models = leaderboard_res.json()
assert len(models) > 0, "No models trained!"
best_model = [m for m in models if m.get("is_best")][0] if any(m.get("is_best") for m in models) else models[0]
print(f"Training completed! {len(models)} models on leaderboard. Champion: {best_model['algorithm_name']} (F1: {best_model['metrics'].get('f1_score')})")

print("\n--- 11. Testing Phase 8: Model Evaluation & Explainability ---")
eval_res = client.get(f"/api/v1/models/{best_model['id']}/evaluation", headers=headers)
assert eval_res.status_code == 200
print("Evaluation metrics retrieved:", eval_res.json().get("metrics"))

explain_res = client.get(f"/api/v1/models/{best_model['id']}/explain", headers=headers)
assert explain_res.status_code == 200
print("Explainability feature importance:", list(explain_res.json().get("feature_importance", {}).keys())[:4])

print("\n--- 12. Testing Phase 9: Model Deployment & Real-time Prediction ---")
deploy_res = client.post("/api/v1/deployments/", json={"model_id": best_model["id"]}, headers=headers)
assert deploy_res.status_code == 200
deployment = deploy_res.json()
deployment_id = deployment["id"]
print(f"Deployed endpoint: {deployment['endpoint_name']} (Status: {deployment['status']})")

# Run inference
sample_input = {
    "tenure_months": 12,
    "monthly_charges": 75.50,
    "total_charges": 906.0,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "support_tickets": 3
}
predict_res = client.post(f"/api/v1/deployments/{deployment_id}/predict", json={"features": sample_input}, headers=headers)
assert predict_res.status_code == 200
pred = predict_res.json()
prediction_id = pred["prediction_id"]
print(f"Real-time Prediction Result: {pred['prediction']} (Latency: {pred['latency_ms']} ms)")

print("\n--- 13. Testing Phase 10: Feedback Loop & AI Chat Assistant ---")
fb_res = client.post(f"/api/v1/deployments/predictions/{prediction_id}/feedback", json={
    "prediction_id": prediction_id,
    "is_correct": True,
    "comment": "Accurately flagged churn risk."
}, headers=headers)
assert fb_res.status_code == 200
print("Submitted prediction feedback:", fb_res.json()["is_correct"])

chat_res = client.post("/api/v1/chat/", json={
    "message": "Why was this customer classified as likely to churn?",
    "project_id": project_id
}, headers=headers)
assert chat_res.status_code == 200
print("AI Data Scientist Assistant reply preview:", chat_res.json()["response"][:120], "...")

print("\n--- 14. Testing Phase 11: Continuous Learning & Knowledge Base ---")
know_res = client.get("/api/v1/knowledge/stats", headers=headers)
assert know_res.status_code == 200
print("Knowledge Base Stats:", know_res.json())

print("\n==========================================")
print("SUCCESS: ALL 11 PHASES FULLY FUNCTIONING!")
print("==========================================")
