"""
Script to enrich test predictions with lifestyle features.

This script joins test predictions with lifestyle features from FE data
to create *_with_lifestyle.csv files for L2 lifestyle analysis.

Usage:
    python -m src.create_lifestyle_enriched_predictions
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enrich_predictions_with_lifestyle(
    predictions_path: Path,
    fe_train_path: Path,
    fe_test_path: Path,
    output_path: Path,
    model_name: str
) -> pd.DataFrame:
    """
    Enrich test predictions with lifestyle features from FE data.
    
    Args:
        predictions_path: Path to test predictions CSV
        fe_train_path: Path to FE train data CSV
        fe_test_path: Path to FE test data CSV
        output_path: Path to save enriched predictions
        model_name: Model name ('rf' or 'tabnet')
        
    Returns:
        Enriched DataFrame with lifestyle features
    """
    logger.info(f"Loading predictions from: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    
    logger.info(f"  Predictions shape: {predictions.shape}")
    logger.info(f"  Columns: {list(predictions.columns)}")
    
    # Load FE data (both train and test to cover all samples)
    fe_data_list = []
    
    if fe_train_path.exists():
        logger.info(f"Loading FE train data from: {fe_train_path}")
        fe_train = pd.read_csv(fe_train_path)
        logger.info(f"  Train data shape: {fe_train.shape}")
        fe_data_list.append(fe_train)
    else:
        logger.warning(f"FE train data not found: {fe_train_path}")
    
    if fe_test_path.exists():
        logger.info(f"Loading FE test data from: {fe_test_path}")
        fe_test = pd.read_csv(fe_test_path)
        logger.info(f"  Test data shape: {fe_test.shape}")
        fe_data_list.append(fe_test)
    else:
        logger.warning(f"FE test data not found: {fe_test_path}")
    
    if not fe_data_list:
        raise FileNotFoundError(f"Neither train nor test FE data found for {model_name}")
    
    # Combine train and test FE data
    fe_data = pd.concat(fe_data_list, ignore_index=True)
    logger.info(f"Combined FE data shape: {fe_data.shape}")
    
    # Check lifestyle columns
    lifestyle_cols = ['smoke', 'alco', 'active', 'lifestyle_risk']
    available_lifestyle = [col for col in lifestyle_cols if col in fe_data.columns]
    logger.info(f"Available lifestyle columns: {available_lifestyle}")
    
    if not available_lifestyle:
        logger.warning("No lifestyle columns found in FE data")
        return predictions
    
    # Check sample_id column
    if 'sample_id' not in predictions.columns:
        raise ValueError(f"Predictions file missing 'sample_id' column: {predictions_path}")
    
    if 'sample_id' not in fe_data.columns:
        raise ValueError(f"FE data missing 'sample_id' column")
    
    # Join predictions with FE data on sample_id
    logger.info("Joining predictions with FE data on sample_id...")
    enriched = predictions.merge(
        fe_data[['sample_id'] + available_lifestyle],
        on='sample_id',
        how='left'
    )
    
    # Check join results
    matched = enriched[available_lifestyle].notna().all(axis=1).sum()
    total = len(enriched)
    logger.info(f"Matched samples with lifestyle data: {matched}/{total} ({matched/total*100:.1f}%)")
    
    if matched < total:
        missing = total - matched
        logger.warning(f"{missing} samples missing lifestyle data")
        missing_ids = enriched[enriched[available_lifestyle].isna().any(axis=1)]['sample_id'].tolist()
        logger.warning(f"Missing sample_ids: {missing_ids[:10]}...")  # Show first 10
    
    # Save enriched predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    logger.info(f"Saved enriched predictions to: {output_path}")
    logger.info(f"  Shape: {enriched.shape}")
    logger.info(f"  Columns: {list(enriched.columns)}")
    
    return enriched


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Enrich test predictions with lifestyle features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--fe_dir',
        type=str,
        default='artifacts/fe',
        help='Feature engineering directory'
    )
    parser.add_argument(
        '--predictions_dir',
        type=str,
        default='artifacts/Interpretation/test_prediction_result',
        help='Directory containing test predictions'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts/Interpretation/test_prediction_result',
        help='Output directory for enriched predictions'
    )
    
    args = parser.parse_args()
    
    fe_dir = Path(args.fe_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ENRICHING TEST PREDICTIONS WITH LIFESTYLE FEATURES")
    logger.info("=" * 60)
    
    # Process RF predictions
    logger.info("\n" + "-" * 60)
    logger.info("Processing Random Forest predictions...")
    logger.info("-" * 60)
    
    rf_predictions = predictions_dir / 'test_predictions_rf.csv'
    rf_train_fe = fe_dir / 'random_forest' / 'Train_data_rf.csv'
    rf_test_fe = fe_dir / 'random_forest' / 'Test_data_rf.csv'
    rf_output = output_dir / 'test_predictions_rf_with_lifestyle.csv'
    
    if rf_predictions.exists():
        try:
            enrich_predictions_with_lifestyle(
                rf_predictions,
                rf_train_fe,
                rf_test_fe,
                rf_output,
                'rf'
            )
            logger.info("✅ RF predictions enriched successfully")
        except Exception as e:
            logger.error(f"❌ Failed to enrich RF predictions: {e}")
    else:
        logger.warning(f"RF predictions file not found: {rf_predictions}")
    
    # Process TabNet predictions
    logger.info("\n" + "-" * 60)
    logger.info("Processing TabNet predictions...")
    logger.info("-" * 60)
    
    tabnet_predictions = predictions_dir / 'test_predictions_tabnet.csv'
    tabnet_train_fe = fe_dir / 'tabnet' / 'Train_data_tabnet.csv'
    tabnet_test_fe = fe_dir / 'tabnet' / 'Test_data_tabnet.csv'
    tabnet_output = output_dir / 'test_predictions_tabnet_with_lifestyle.csv'
    
    if tabnet_predictions.exists():
        try:
            enrich_predictions_with_lifestyle(
                tabnet_predictions,
                tabnet_train_fe,
                tabnet_test_fe,
                tabnet_output,
                'tabnet'
            )
            logger.info("✅ TabNet predictions enriched successfully")
        except Exception as e:
            logger.error(f"❌ Failed to enrich TabNet predictions: {e}")
    else:
        logger.warning(f"TabNet predictions file not found: {tabnet_predictions}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

