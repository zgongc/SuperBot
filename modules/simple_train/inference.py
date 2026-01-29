
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from modules.simple_train.models.entry_model import EntryModel
from modules.simple_train.core.feature_extractor import FeatureExtractor
from core.logger_engine import get_logger

logger = get_logger(__name__)

class SimpleTrainPredictor:
    """
    Predictor wrapper for SimpleTrain EntryModel.
    Integrates with BacktestEngine's AI interface.
    """
    def __init__(self, model_path: str, strategy_name: str = "simple_rsi"):
        self.model_path = Path(model_path)
        self.strategy_name = strategy_name
        self.model = None
        self.feature_names = [] # Features expected by the model
        
        # Initialize FeatureExtractor to calculate derived features (momentum, etc.)
        self.feature_extractor = FeatureExtractor(strategy_name=strategy_name)
        
        self._load_model()

    def _load_model(self):
        """Load the trained EntryModel and Metadata"""
        try:
            # FIX: EntryModel does not take strategy_name
            self.model = EntryModel()
            self.model.load(str(self.model_path))
            logger.info(f"✅ SimpleTrain model loaded: {self.model_path}")
            
            # Load metadata to get feature names
            # Model path is like .../entry_model_timestamp/model.pkl
            # Metadata is in .../entry_model_timestamp/metadata.yaml
            metadata_path = self.model_path.parent / "metadata.yaml"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        meta = yaml.safe_load(f)
                        self.feature_names = meta.get('feature_names', [])
                        logger.info(f"📋 Loaded {len(self.feature_names)} features from metadata")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse metadata: {e}")
            else:
                logger.warning(f"⚠️ Metadata not found at {metadata_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load SimpleTrain model: {e}")
            raise

    def predict_batch(self, features_df: pd.DataFrame) -> dict:
        """
        Predict probabilities for a batch of data.
        
        Args:
            features_df: DataFrame with ALL features (already extracted)
            
        Returns:
            dict: {index: probability} mapping
        """
        if self.model is None:
            return {}
            
        try:
            # 1. Feature Extraction (Enrichment - Base Timeframe)
            # Calculate derived features (momentum, distance, etc) that might be missing
            enriched_df = self.feature_extractor.extract(features_df)
            
            # 2. MTF Derived Feature Extraction (Critical for MTF Models)
            # We logic is preserved here although current model might not use MTF features.
            # It's good to have for future MTF models.
            
            mtf_suffixes = []
            for col in enriched_df.columns:
                if col.startswith('close_') and col != 'close_pct':
                    suffix = col.replace('close_', '')
                    if suffix not in mtf_suffixes:
                        mtf_suffixes.append(suffix)
            
            for suffix in mtf_suffixes:
                temp_df = pd.DataFrame(index=enriched_df.index)
                has_ohlc = True
                for core_col in ['open', 'high', 'low', 'close', 'volume']:
                    mtf_col = f"{core_col}_{suffix}"
                    if mtf_col in enriched_df.columns:
                        temp_df[core_col] = enriched_df[mtf_col]
                    else:
                        has_ohlc = False
                        break
                
                if not has_ohlc:
                    continue
                    
                for col in enriched_df.columns:
                    if col.endswith(f"_{suffix}"):
                        base_name = col[:-len(suffix)-1]
                        if base_name not in ['open', 'high', 'low', 'close', 'volume']:
                            temp_df[base_name] = enriched_df[col]
                            
                temp_enriched = self.feature_extractor.extract(temp_df)
                
                for col in temp_enriched.columns:
                    if col in temp_df.columns:
                        continue
                    target_col = f"{col}_{suffix}"
                    enriched_df[target_col] = temp_enriched[col]

            # 3. Predict preparation
            # Remove non-numeric columns like timestamps or strings
            X_df = enriched_df.select_dtypes(include=[np.number])
            
            # --- FEATURE ALIGNMENT ---
            # Detect what features the model actually expects
            model_features = None
            
            # 0. Priority: Metadata (Loaded in _load_model)
            if self.feature_names:
                model_features = self.feature_names
            
            # 1. Scikit-Learn / XGBoost Sklearn API
            elif self.model and hasattr(self.model, 'model'):
                raw_model = self.model.model
                if hasattr(raw_model, "feature_names_in_"):
                    model_features = raw_model.feature_names_in_
                elif hasattr(raw_model, "get_booster"):
                    try:
                        model_features = raw_model.get_booster().feature_names
                    except:
                        pass
                elif hasattr(raw_model, "feature_name_"):
                    model_features = raw_model.feature_name_
            
            # Apply alignment if we found expected features
            if model_features is not None and len(model_features) > 0:
                # Check for missing features
                missing = [f for f in model_features if f not in X_df.columns]
                if missing:
                    logger.warning(f"⚠️ Missing features for AI model: {missing}. Filling with 0.")
                    for f in missing:
                        X_df[f] = 0.0
                
                # Reorder columns EXACTLY as model expects
                X_df = X_df[model_features]
            
            # Fallback: Drop explicit non-features if alignment failed
            else:
                cols_to_drop = [c for c in X_df.columns if c in ['open_time', 'close_time', 'id', 'timestamp']]
                if cols_to_drop:
                    X_df = X_df.drop(columns=cols_to_drop)
                logger.warning("⚠️ Could not detect model feature names. Sending all numeric columns.")

            probs = self.model.predict_proba(X_df)
            
            # 4. Convert to dict {index: prob}
            return dict(zip(enriched_df.index, probs))
            
        except Exception as e:
            logger.error(f"❌ AI Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {}
