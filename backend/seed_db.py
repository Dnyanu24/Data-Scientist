import sys
from app.database.session import SyncSessionLocal
from app.services.seed_data import seed_algorithm_knowledge
from app.models.models import AlgorithmKnowledge

db = SyncSessionLocal()
try:
    seed_algorithm_knowledge(db)
    count = db.query(AlgorithmKnowledge).count()
    print(f"Database AlgorithmKnowledge count: {count}", flush=True)
finally:
    db.close()
