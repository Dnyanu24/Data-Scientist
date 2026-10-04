from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ─── Auth ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: str = Field(..., min_length=5)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenRefresh(BaseModel):
    refresh_token: str


# ─── Projects ────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    business_goal: Optional[str] = None
    target_column: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    business_goal: Optional[str] = None
    target_column: Optional[str] = None
    problem_type: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    business_goal: Optional[str]
    problem_type: Optional[str]
    target_column: Optional[str]
    phase: str
    phase_progress: float
    phase_message: Optional[str]
    owner_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int


# ─── Datasets ────────────────────────────────────────────────────────────────

class DatasetResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    original_filename: str
    file_size: Optional[int]
    file_type: str
    num_rows: Optional[int]
    num_columns: Optional[int]
    columns_info: Optional[List[Dict[str, Any]]]
    schema_info: Optional[Dict[str, Any]]
    profile_report: Optional[Dict[str, Any]]
    data_quality: Optional[Dict[str, Any]]
    target_analysis: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Fingerprint ──────────────────────────────────────────────────────────────

class FingerprintResponse(BaseModel):
    id: int
    dataset_id: int
    project_id: int
    num_rows: Optional[int]
    num_columns: Optional[int]
    num_numeric: Optional[int]
    num_categorical: Optional[int]
    num_text: Optional[int]
    num_datetime: Optional[int]
    missing_ratio: Optional[float]
    class_imbalance_ratio: Optional[float]
    avg_correlation: Optional[float]
    feature_to_sample_ratio: Optional[float]
    outlier_ratio: Optional[float]
    numeric_skewness_mean: Optional[float]
    problem_type: Optional[str]
    fingerprint_data: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


DatasetFingerprintResponse = FingerprintResponse


class SimilarDatasetResponse(BaseModel):
    fingerprint: FingerprintResponse
    similarity_score: float
    experiment_results: Optional[List[Dict[str, Any]]] = None


# ─── Recommendations ─────────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    id: int
    project_id: int
    algorithm_name: str
    recommendation_score: float
    confidence: float
    training_priority: int
    meta_learning_score: float
    historical_score: float
    rule_based_score: float
    compatibility_score: float
    user_requirement_score: float
    reasoning: Optional[str]
    training_stage: str
    created_at: datetime

    class Config:
        from_attributes = True


AlgorithmRecommendationResponse = RecommendationResponse


# ─── Preprocessing ───────────────────────────────────────────────────────────

class PreprocessingStep(BaseModel):
    step: str
    strategy: str
    params: Optional[Dict[str, Any]] = None


class PreprocessingConfigResponse(BaseModel):
    id: int
    project_id: int
    pipeline_steps: Optional[List[Dict[str, Any]]]
    feature_columns: Optional[List[str]]
    target_column: Optional[str]
    applied: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PreprocessingConfigUpdate(BaseModel):
    pipeline_steps: Optional[List[Dict[str, Any]]] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None


# ─── Training ────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    algorithms: Optional[List[str]] = None  # None = use recommendations
    max_trials_stage1: int = 15
    max_trials_stage2: int = 40
    max_trials_stage3: int = 100
    create_ensembles: bool = True


class TrainingProgressResponse(BaseModel):
    project_id: int
    current_stage: str
    overall_progress: float
    current_algorithm: Optional[str]
    completed_algorithms: List[str]
    results: List[Dict[str, Any]]
    message: Optional[str]


# ─── Trained Models ──────────────────────────────────────────────────────────

class TrainedModelResponse(BaseModel):
    id: int
    project_id: int
    experiment_id: Optional[int]
    name: str
    algorithm_name: str
    version: int
    status: str
    metrics: Optional[Dict[str, Any]]
    hyperparameters: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, Any]]
    explainability: Optional[Dict[str, Any]]
    training_stage: Optional[str]
    is_best: bool
    is_ensemble: bool
    ensemble_type: Optional[str]
    file_size: Optional[int]
    training_time: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Experiments ──────────────────────────────────────────────────────────────

class ExperimentResponse(BaseModel):
    id: int
    project_id: int
    mlflow_run_id: Optional[str]
    name: str
    algorithm_name: str
    hyperparameters: Optional[Dict[str, Any]]
    training_stage: Optional[str]
    metrics: Optional[Dict[str, Any]]
    training_time: Optional[float]
    num_trials: Optional[int]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─── Deployment ───────────────────────────────────────────────────────────────

class DeployRequest(BaseModel):
    model_id: int
    endpoint_name: Optional[str] = None


class DeploymentResponse(BaseModel):
    id: int
    project_id: int
    model_id: int
    endpoint_name: str
    status: str
    version: int
    request_count: int
    avg_latency_ms: Optional[float]
    last_prediction_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Predictions ──────────────────────────────────────────────────────────────

class PredictionCreate(BaseModel):
    features: Any  # dict or list of dicts


class PredictRequest(BaseModel):
    data: Dict[str, Any]  # single row
    

class BatchPredictRequest(BaseModel):
    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    id: int
    project_id: int
    deployment_id: int
    input_data: Dict[str, Any]
    prediction: Any
    probability: Optional[Any]
    latency_ms: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


# ─── Feedback ─────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    prediction_id: Optional[int] = None
    is_correct: Optional[bool] = None
    correct_value: Optional[str] = None
    comment: Optional[str] = None


PredictionFeedbackCreate = FeedbackCreate


class FeedbackResponse(BaseModel):
    id: int
    prediction_id: int
    user_id: int
    is_correct: Optional[bool]
    correct_value: Optional[str]
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


PredictionFeedbackResponse = FeedbackResponse


# ─── Drift ────────────────────────────────────────────────────────────────────

class DriftRecordResponse(BaseModel):
    id: int
    deployment_id: int
    drift_type: str
    feature_name: Optional[str]
    drift_score: float
    is_drifted: bool
    details: Optional[Dict[str, Any]]
    window_start: Optional[datetime]
    window_end: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Knowledge ────────────────────────────────────────────────────────────────

class AlgorithmKnowledgeResponse(BaseModel):
    id: int
    name: str
    display_name: str
    category: Optional[str]
    library: Optional[str]
    supports_classification: bool
    supports_regression: bool
    supports_multiclass: bool
    handles_missing: bool
    handles_categorical: bool
    handles_imbalanced: bool
    training_speed: Optional[str]
    prediction_speed: Optional[str]
    memory_usage: Optional[str]
    scalability: Optional[str]
    description: Optional[str]
    strengths: Optional[List[str]]
    weaknesses: Optional[List[str]]

    class Config:
        from_attributes = True


class ExperienceResponse(BaseModel):
    id: int
    project_id: Optional[int]
    problem_type: Optional[str]
    dataset_size: Optional[str]
    num_features: Optional[int]
    best_algorithm: Optional[str]
    best_metrics: Optional[Dict[str, Any]]
    all_results: Optional[List[Dict[str, Any]]]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Task Progress & Background Tasks ─────────────────────────────────────────

class TaskProgressResponse(BaseModel):
    task_id: int
    celery_task_id: Optional[str]
    task_type: str
    status: str
    progress: float
    message: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]

    class Config:
        from_attributes = True


class BackgroundTaskResponse(BaseModel):
    id: int
    project_id: Optional[int] = None
    celery_task_id: Optional[str] = None
    task_type: str
    status: str
    progress: float
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─── Goal Understanding ───────────────────────────────────────────────────────

class GoalUnderstandingRequest(BaseModel):
    business_goal: Optional[str] = None


class GoalUnderstandingResponse(BaseModel):
    problem_type: str
    recommended_target_column: Optional[str] = None
    recommended_metric: str
    business_objective_summary: str
    business_success_criteria: str
    key_risks_and_considerations: List[str]


# ─── Chat Assistant ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    project_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = None
    code_snippet: Optional[str] = None
