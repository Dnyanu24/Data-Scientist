"""Script to clean up automated demo/test 'Customer Churn Prediction' projects and artifacts,
retaining the user's original data: 'salary prediction', 'house price prediction', and 'loan',
along with the system knowledge base (62 algorithms & 16 meta-learning records).
"""
import os
import shutil
import sqlite3
import glob

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ai_datascientist.db"))
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

print(f"Connecting to database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Find test projects
cursor.execute("SELECT id, name FROM projects WHERE name = 'Customer Churn Prediction'")
demo_projects = cursor.fetchall()
demo_project_ids = [p[0] for p in demo_projects]
print(f"Found {len(demo_project_ids)} demo projects to remove: {demo_project_ids}")

if not demo_project_ids:
    print("No demo projects found. Nothing to remove.")
    conn.close()
    exit(0)

# Build comma separated string for SQL IN clauses
placeholders = ",".join(str(pid) for pid in demo_project_ids)

# 1. Get Datasets to remove
cursor.execute(f"SELECT id, filename, file_path FROM datasets WHERE project_id IN ({placeholders})")
demo_datasets = cursor.fetchall()
demo_dataset_ids = [d[0] for d in demo_datasets]
print(f"Found {len(demo_dataset_ids)} demo datasets.")

# 2. Get Models to remove
cursor.execute(f"SELECT id FROM trained_models WHERE project_id IN ({placeholders})")
demo_models = cursor.fetchall()
demo_model_ids = [m[0] for m in demo_models]
print(f"Found {len(demo_model_ids)} demo models.")

# 3. Get Deployments to remove
demo_deployment_ids = []
if demo_model_ids:
    model_placeholders = ",".join(str(mid) for mid in demo_model_ids)
    cursor.execute(f"SELECT id FROM deployments WHERE model_id IN ({model_placeholders})")
    demo_deployments = cursor.fetchall()
    demo_deployment_ids = [d[0] for d in demo_deployments]
print(f"Found {len(demo_deployment_ids)} demo deployments.")

# Now delete in correct foreign key order
if demo_deployment_ids:
    dep_placeholders = ",".join(str(did) for did in demo_deployment_ids)
    cursor.execute(f"DELETE FROM prediction_feedbacks WHERE prediction_id IN (SELECT id FROM predictions WHERE deployment_id IN ({dep_placeholders}))")
    cursor.execute(f"DELETE FROM drift_records WHERE deployment_id IN ({dep_placeholders})")
    cursor.execute(f"DELETE FROM predictions WHERE deployment_id IN ({dep_placeholders})")
    cursor.execute(f"DELETE FROM deployments WHERE id IN ({dep_placeholders})")
    print(f"Deleted deployments and related records.")

if demo_model_ids:
    cursor.execute(f"DELETE FROM trained_models WHERE project_id IN ({placeholders})")
    print(f"Deleted {len(demo_model_ids)} trained models.")

cursor.execute(f"DELETE FROM experiments WHERE project_id IN ({placeholders})")
print("Deleted experiments.")

if demo_dataset_ids:
    ds_placeholders = ",".join(str(did) for did in demo_dataset_ids)
    cursor.execute(f"DELETE FROM dataset_fingerprints WHERE dataset_id IN ({ds_placeholders})")
    cursor.execute(f"DELETE FROM algorithm_recommendations WHERE project_id IN ({placeholders})")
    cursor.execute(f"DELETE FROM preprocessing_configs WHERE project_id IN ({placeholders})")
    cursor.execute(f"DELETE FROM datasets WHERE project_id IN ({placeholders})")
    print(f"Deleted {len(demo_dataset_ids)} datasets and associated fingerprints/recommendations.")

cursor.execute(f"DELETE FROM background_tasks WHERE project_id IN ({placeholders})")
cursor.execute(f"DELETE FROM projects WHERE id IN ({placeholders})")
print(f"Deleted {len(demo_project_ids)} projects.")

conn.commit()
cursor.execute("VACUUM")
cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
conn.close()
print("Database cleanup committed and vacuumed.")

# 4. Clean up files on disk: upload files for churn_data
for upload_dir in [os.path.join(BACKEND_DIR, "uploads"), os.path.join(ROOT_DIR, "uploads")]:
    if os.path.exists(upload_dir):
        for pattern in ["*churn_data.csv", "*churn*.csv"]:
            for f in glob.glob(os.path.join(upload_dir, pattern)):
                try:
                    os.remove(f)
                    print(f"Removed upload file: {f}")
                except Exception as e:
                    print(f"Error removing {f}: {e}")

# 5. Clean up model_store directories for demo projects
for store_dir in [os.path.join(BACKEND_DIR, "model_store"), os.path.join(ROOT_DIR, "model_store")]:
    if os.path.exists(store_dir):
        for pid in demo_project_ids:
            p_dir = os.path.join(store_dir, str(pid))
            if os.path.exists(p_dir):
                try:
                    shutil.rmtree(p_dir)
                    print(f"Removed model store directory: {p_dir}")
                except Exception as e:
                    print(f"Error removing {p_dir}: {e}")

# Also check root database Data-Scientist/ai_datascientist.db if present
root_db = os.path.join(ROOT_DIR, "ai_datascientist.db")
if os.path.exists(root_db):
    try:
        shutil.copy2(DB_PATH, root_db)
        print("Synchronized root ai_datascientist.db with cleaned backend database.")
    except Exception as e:
        print(f"Note: Could not sync root db: {e}")

print("\n--- CLEANUP COMPLETE ---")
