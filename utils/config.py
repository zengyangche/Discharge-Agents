"""
Configuration file: defines models, paths, and parameters
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data path configuration
DATA_DIR = PROJECT_ROOT / "data"

# Dataset selection (can be overridden via DATASET_SPLIT environment variable)
# Options: "train", "valid", "test_phase_1", "test_phase_2"
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "test_phase_1")

# Label file paths (fixed in the data root directory)
TRAIN_DATA_PATH = DATA_DIR / "discharge_target_train.csv"
TEST_DATA_PATH = DATA_DIR / "discharge_target_test.csv"

# Determine raw data file paths based on dataset type
if DATASET_SPLIT == "train":
    # Training set is in the root directory (flat structure)
    DIAGNOSIS_PATH = DATA_DIR / "diagnosis.csv"
    EDSTAYS_PATH = DATA_DIR / "edstays.csv"
    TRIAGE_PATH = DATA_DIR / "triage.csv"
    DISCHARGE_PATH = DATA_DIR / "discharge.csv"
elif DATASET_SPLIT in ["test_phase_1", "test_phase_2", "valid"]:
    # Test and validation sets are in subdirectories (nested structure)
    DATA_BASE = DATA_DIR / DATASET_SPLIT
    DIAGNOSIS_PATH = DATA_BASE / "diagnosis" / "diagnosis.csv"
    EDSTAYS_PATH = DATA_BASE / "edstays" / "edstays.csv"
    TRIAGE_PATH = DATA_BASE / "triage" / "triage.csv"
    DISCHARGE_PATH = DATA_BASE / "discharge" / "discharge.csv"
else:
    # Fallback strategy: use files from the root directory
    DIAGNOSIS_PATH = DATA_DIR / "diagnosis.csv"
    EDSTAYS_PATH = DATA_DIR / "edstays.csv"
    TRIAGE_PATH = DATA_DIR / "triage.csv"
    DISCHARGE_PATH = DATA_DIR / "discharge.csv"

# Output path
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration (unified OpenAI-compatible API, differentiated by model name)
# Medical text generation optimization: moderate temperature balances accuracy and diversity; penalty added to reduce repetition
MODEL_ZOO = {
    "gpt-4o": {
        "provider": "openai",  # Unified openai provider; differentiated by base_url and model name
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 4096,
        "frequency_penalty": 0.5,  # Reduce repetitive content
    },
    "claude-3-5-sonnet-latest": {
        "provider": "openai",
        "model_name": "claude-3-5-sonnet-latest",
        "temperature": 0.0,
        "max_tokens": 4096,
        "frequency_penalty": 0.5,
    },
    "deepseek-v3": {
        "provider": "openai",  # Unified openai provider
        "model_name": "deepseek-v3",  # Use correct model name
        "temperature": 0.0,
        "max_tokens": 4096,
        "frequency_penalty": 0.5,
    },
    "gemini-3.5-flash": {
        "provider": "openai",
        "model_name": "gemini-3.5-flash",
        "temperature": 0.0,
        "max_tokens": 4096,  # Needs enough space for reasoning tokens + text tokens
        "frequency_penalty": 0.5,
    },
    "grok-4.3": {
        "provider": "openai",
        "model_name": "grok-4.3",
        "temperature": 0.0,
        "max_tokens": 4096,
        # frequency_penalty not supported, not passed
    }
}

# Also exported as MODELS for compatibility
MODELS = MODEL_ZOO

# Verification configuration
VERIFICATION_CONFIG = {
    "fact_check": {
        "enabled": True,
        "similarity_threshold": 0.8,
        "weight": 0.3,
    },
    "style_check": {
        "bhc_embedding_similarity_threshold": 0.7,
        "bhc_weight": 0.2,
        "di_readability_threshold": 60.0,  # Flesch Reading Ease (0-100, higher is easier)
        "di_readability_weight": 0.15,
        "di_terminology_residue_threshold": 0.1,  # 10% jargon allowed
        "di_terminology_weight": 0.0,  # Disable jargon residue check (set to 0)
    },
    "logic_check": {
        "contradiction_threshold": 0.8,
        "weight": 0.1,
    },
    "clinical_logic_check": {
        "enabled": True,
        "weight": 0.1,
    }
}

# Association rules configuration
ASSOCIATION_RULES_CONFIG = {
    "min_support": 0.01,
    "min_confidence": 0.85,
    "max_rules": 1000,
}

# Medical terminology list (used for jargon residue detection in style verification)
# Note: jargon residue check is disabled (weight set to 0); this list is kept only for code compatibility
MEDICAL_TERMINOLOGY = []

# Unified API configuration (using integrated platform)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "xxxxxx")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.bianxie.ai/v1")

# Other API configurations retained for compatibility (all use the unified API above)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QWEN_API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "")
GEMINI_API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "")

# Multi-agent framework configuration
MULTI_AGENT_CONFIG = {
    # Segmentation agent configuration
    "segmentation": {
        "model_name": "gpt-4o",  # Model used by the segmentation agent
        "temperature": 0.0,
    },
    # Generation agent configuration
    "generation": {
        # List of models used by the generation agent (medical text generation, multiple models for higher accuracy)
        "model_list": [
            "gpt-4o",
            "claude-3-5-sonnet-latest",
            "deepseek-v3",
            "gemini-3.5-flash",
            "grok-4.3",
        ],
        # Medical text generation optimization parameters
        "temperature": 0.0,
        "frequency_penalty": 0.5,  # Reduce repetitive content
        # "min_tokens": 200,  # Minimum output token count
        "max_tokens": 4096,  # Increased to 4096 to accommodate full output for complex cases
    },
    # Output directory
    "output_dir": "outputs/results/multi_agent/prompt_v2",
    # Verification agent configuration
    "verification": {
        "enabled": True,
        "weights": {
            "fact": 0.5,
            "logic": 0.3,
            "style": 0.2
        },
        "thresholds": {
            "fact": 0.6,
            "logic": 0.6,
            "style": 0.5
        },
    },
    # Knowledge base directory
    "knowledge_base_dir": "outputs/knowledge_base",
    # Summary agent configuration
    "summary": {
        # List of models participating in discussion (defaults to the generation agent model list)
        "discussion_models": [
            "gpt-4o",
            "claude-3-5-sonnet-latest",
            "deepseek-v3",
            "gemini-3.5-flash",
            "grok-4.3",
        ],
        # Discussion parameters (optimized)
        "discussion_temperature": 0.0,
        "discussion_max_tokens": 2048,
        "max_voting_rounds": 3  # Maximum voting rounds (multiple rounds until passed or max reached)
    }
}
