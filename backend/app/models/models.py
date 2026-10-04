import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, JSON,
    ForeignKey, Enum as SAEnum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
import enum

# Conditional pgvector import - falls back to JSON for SQLite
try:
    from pgvector.sqlalchemy import Vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    Vector = lambda dim: JSON  # Fallback: store as JSON array

from app.database.session import Base


# ─── Enums ───────────────────────────────────────────────────────────────────

class ProblemType(str, enum.Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"


class PipelinePhase(str, enum.Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROFILING = "profiling"
    PROFILED = "profiled"
    FINGERPRINTING = "fingerprinting"
    FINGERPRINTED = "fingerprinted"
    RECOMMENDING = "recommending"
    RECOMMENDED = "recommended"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    REGISTERED = "registered"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


class TrainingStage(str, enum.Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    ENSEMBLE = "ensemble"


# ─── Users ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    feedbacks = relationship("PredictionFeedback", back_populates="user")


# ─── Projects ────────────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    business_goal = Column(Text, nullable=True)
    problem_type = Column(SAEnum(ProblemType), nullable=True)
    target_column = Column(String(255), nullable=True)
    phase = Column(SAEnum(PipelinePhase), default=PipelinePhase.CREATED)
    phase_progress = Column(Float, default=0.0)
    phase_message = Column(String(500), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    trained_models = relationship("TrainedModel", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="project", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="project", cascade="all, delete-orphan")
    recommendations = relationship("AlgorithmRecommendation", back_populates="project", cascade="all, delete-orphan")
    preprocessing_config = relationship("PreprocessingConfig", back_populates="project", uselist=False, cascade="all, delete-orphan")


# ─── Datasets ────────────────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String(50), nullable=False)  # csv, xlsx
    num_rows = Column(Integer, nullable=True)
    num_columns = Column(Integer, nullable=True)
    columns_info = Column(JSON, nullable=True)  # [{name, dtype, nunique, missing, sample}]
    schema_info = Column(JSON, nullable=True)   # detected types, categories
    profile_report = Column(JSON, nullable=True)  # full profiling result
    data_quality = Column(JSON, nullable=True)   # quality scores, issues
    target_analysis = Column(JSON, nullable=True) # target column analysis
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="datasets")
    fingerprint = relationship("DatasetFingerprint", back_populates="dataset", uselist=False, cascade="all, delete-orphan")


# ─── Dataset Fingerprint ─────────────────────────────────────────────────────

class DatasetFingerprint(Base):
    __tablename__ = "dataset_fingerprints"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), unique=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Characteristics
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    num_numeric = Column(Integer)
    num_categorical = Column(Integer)
    num_text = Column(Integer)
    num_datetime = Column(Integer)
    missing_ratio = Column(Float)
    class_imbalance_ratio = Column(Float, nullable=True)
    avg_correlation = Column(Float, nullable=True)
    feature_to_sample_ratio = Column(Float)
    outlier_ratio = Column(Float, nullable=True)
    numeric_skewness_mean = Column(Float, nullable=True)
    problem_type = Column(SAEnum(ProblemType), nullable=True)

    # Full fingerprint JSON
    fingerprint_data = Column(JSON, nullable=True)

    # Embedding for similarity search
    embedding = Column(Vector(128), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="fingerprint")

    __table_args__ = (
        Index('ix_fingerprint_embedding', embedding, postgresql_using='ivfflat'),
    )


# ─── Algorithm Knowledge Base ────────────────────────────────────────────────

class AlgorithmKnowledge(Base):
    __tablename__ = "algorithm_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    category = Column(String(100))  # tree, linear, ensemble, svm, nn, etc.
    library = Column(String(100))   # sklearn, xgboost, lightgbm, catboost
    class_path = Column(String(500))  # e.g. "sklearn.ensemble.RandomForestClassifier"

    # Capabilities
    supports_classification = Column(Boolean, default=False)
    supports_regression = Column(Boolean, default=False)
    supports_multiclass = Column(Boolean, default=False)
    handles_missing = Column(Boolean, default=False)
    handles_categorical = Column(Boolean, default=False)
    handles_text = Column(Boolean, default=False)
    handles_imbalanced = Column(Boolean, default=False)
    supports_feature_importance = Column(Boolean, default=False)
    is_ensemble = Column(Boolean, default=False)

    # Performance characteristics
    training_speed = Column(String(20))  # fast, medium, slow
    prediction_speed = Column(String(20))
    memory_usage = Column(String(20))  # low, medium, high
    scalability = Column(String(20))  # small, medium, large

    # Hyperparameter space
    default_params = Column(JSON, nullable=True)
    search_space = Column(JSON, nullable=True)

    # Meta info
    description = Column(Text, nullable=True)
    strengths = Column(JSON, nullable=True)  # list of strings
    weaknesses = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# ─── Algorithm Recommendations ───────────────────────────────────────────────

class AlgorithmRecommendation(Base):
    __tablename__ = "algorithm_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    algorithm_name = Column(String(100), nullable=False)
    recommendation_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    training_priority = Column(Integer, nullable=False)

    # Score breakdown
    meta_learning_score = Column(Float, default=0.0)
    historical_score = Column(Float, default=0.0)
    rule_based_score = Column(Float, default=0.0)
    compatibility_score = Column(Float, default=0.0)
    user_requirement_score = Column(Float, default=0.0)

    reasoning = Column(Text, nullable=True)
    training_stage = Column(SAEnum(TrainingStage), default=TrainingStage.STAGE_1)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="recommendations")


# ─── Preprocessing Config ────────────────────────────────────────────────────

class PreprocessingConfig(Base):
    __tablename__ = "preprocessing_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), unique=True, nullable=False)

    # Pipeline steps as JSON
    pipeline_steps = Column(JSON, nullable=True)
    # e.g. [
    #   {"step": "impute_numeric", "strategy": "median"},
    #   {"step": "impute_categorical", "strategy": "mode"},
    #   {"step": "encode_categorical", "strategy": "target_encoding"},
    #   {"step": "scale_numeric", "strategy": "standard"},
    #   {"step": "handle_outliers", "strategy": "clip"},
    #   {"step": "feature_selection", "strategy": "mutual_info", "k": 20},
    # ]

    feature_columns = Column(JSON, nullable=True)
    target_column = Column(String(255), nullable=True)
    applied = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="preprocessing_config")


# ─── Experiments ──────────────────────────────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    mlflow_experiment_id = Column(String(100), nullable=True)
    mlflow_run_id = Column(String(100), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Fingerprint for meta-learning
    dataset_fingerprint_id = Column(Integer, ForeignKey("dataset_fingerprints.id"), nullable=True)

    algorithm_name = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    training_stage = Column(SAEnum(TrainingStage), nullable=True)

    # Results
    metrics = Column(JSON, nullable=True)  # {accuracy, f1, precision, recall, mse, r2, etc.}
    training_time = Column(Float, nullable=True)
    num_trials = Column(Integer, nullable=True)

    status = Column(String(50), default="running")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="experiments")
    fingerprint = relationship("DatasetFingerprint")


# ─── Trained Models ──────────────────────────────────────────────────────────

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

    name = Column(String(255), nullable=False)
    algorithm_name = Column(String(100), nullable=False)
    version = Column(Integer, default=1)
    status = Column(SAEnum(ModelStatus), default=ModelStatus.TRAINING)

    # Storage
    model_path = Column(String(1000), nullable=True)
    pipeline_path = Column(String(1000), nullable=True)  # preprocessing pipeline

    # Performance
    metrics = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    training_stage = Column(SAEnum(TrainingStage), nullable=True)

    # Meta
    is_best = Column(Boolean, default=False)
    is_ensemble = Column(Boolean, default=False)
    ensemble_type = Column(String(50), nullable=True)  # voting, stacking, blending
    ensemble_members = Column(JSON, nullable=True)  # list of model ids

    file_size = Column(Integer, nullable=True)
    training_time = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="trained_models")
    experiment = relationship("Experiment")
    deployment = relationship("Deployment", back_populates="model", uselist=False)


# ─── Deployments ──────────────────────────────────────────────────────────────

class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("trained_models.id"), unique=True, nullable=False)

    endpoint_name = Column(String(255), nullable=False)
    status = Column(SAEnum(DeploymentStatus), default=DeploymentStatus.PENDING)
    version = Column(Integer, default=1)

    # Config
    config = Column(JSON, nullable=True)

    # Monitoring
    request_count = Column(Integer, default=0)
    avg_latency_ms = Column(Float, nullable=True)
    last_prediction_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="deployments")
    model = relationship("TrainedModel", back_populates="deployment")
    predictions = relationship("Prediction", back_populates="deployment")


# ─── Predictions ──────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=False)

    input_data = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    probability = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="predictions")
    deployment = relationship("Deployment", back_populates="predictions")
    feedback = relationship("PredictionFeedback", back_populates="prediction", uselist=False)


# ─── Prediction Feedback ─────────────────────────────────────────────────────

class PredictionFeedback(Base):
    __tablename__ = "prediction_feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    is_correct = Column(Boolean, nullable=True)
    correct_value = Column(String(500), nullable=True)
    comment = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    prediction = relationship("Prediction", back_populates="feedback")
    user = relationship("User", back_populates="feedbacks")


# ─── Drift Monitoring ────────────────────────────────────────────────────────

class DriftRecord(Base):
    __tablename__ = "drift_records"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=False)

    drift_type = Column(String(50))  # data_drift, concept_drift, prediction_drift
    feature_name = Column(String(255), nullable=True)
    drift_score = Column(Float, nullable=False)
    is_drifted = Column(Boolean, default=False)
    details = Column(JSON, nullable=True)
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# ─── Knowledge / Experience ──────────────────────────────────────────────────

class ExperienceRecord(Base):
    __tablename__ = "experience_records"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Dataset characteristics summary
    problem_type = Column(SAEnum(ProblemType), nullable=True)
    dataset_size = Column(String(50), nullable=True)  # small, medium, large
    num_features = Column(Integer, nullable=True)
    feature_types = Column(JSON, nullable=True)  # {numeric: 10, categorical: 5, ...}
    data_quality_score = Column(Float, nullable=True)

    # What worked
    best_algorithm = Column(String(100), nullable=True)
    best_metrics = Column(JSON, nullable=True)
    best_hyperparameters = Column(JSON, nullable=True)
    preprocessing_steps = Column(JSON, nullable=True)

    # Fingerprint embedding for retrieval
    fingerprint_embedding = Column(Vector(128), nullable=True)

    # All tried algorithms
    all_results = Column(JSON, nullable=True)  # [{algo, metrics, rank}]

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# ─── Background Tasks ────────────────────────────────────────────────────────

class BackgroundTask(Base):
    __tablename__ = "background_tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    celery_task_id = Column(String(255), nullable=True, index=True)

    task_type = Column(String(100), nullable=False)  # profiling, training, etc.
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    message = Column(String(500), nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
