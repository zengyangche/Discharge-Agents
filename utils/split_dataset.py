"""
Split discharge_target.csv by BHC and DI length distribution:
sample 300 test cases with matching distribution; remainder is training set.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def stratified_sample_by_length(file_path, n_test=300, columns=['brief_hospital_course', 'discharge_instructions']):
    """
    Stratified sampling by length across multiple columns.
    Keeps test-set length distribution similar to the population.
    """
    print("Reading data...")
    # Read data file
    df = pd.read_csv(file_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Compute length
    for col in columns:
        length_col = f'{col}_length'
        df[length_col] = df[col].fillna('').astype(str).str.len()
        print(f"\n{col} length stats:")
        print(f"  Min: {df[length_col].min()}")
        print(f"  Max: {df[length_col].max()}")
        print(f"  Mean: {df[length_col].mean():.2f}")
        print(f"  Median: {df[length_col].median():.2f}")
        print(f"  Std: {df[length_col].std():.2f}")
    
    # Remove rows with missing values (if any)
    original_len = len(df)
    df = df.dropna(subset=columns)
    print(f"\nRows after removing missing values: {len(df)} (removed {original_len - len(df)} rows)")
    
    if len(df) < n_test:
        raise ValueError(f"Data row count ({len(df)}) is less than test set size ({n_test})")
    
    # Create combined length feature for stratification
    # Normalize and sum two length fields to create strata
    length_cols = [f'{col}_length' for col in columns]
    
    # Use quantile binning, combining both length fields
    n_bins = min(20, len(df) // 100)  # Ensure each bin has sufficient data
    n_bins = max(5, n_bins)  # At least 5 bins
    
    # Create combined length score (after normalization)
    length_features = []
    for col in length_cols:
        col_std = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
        length_features.append(col_std.values)  # Convert to numpy array
    
    # Sum multiple normalized length features
    combined_length = np.sum(length_features, axis=0)
    df['combined_length_score'] = combined_length
    
    # Create strata based on combined score
    df['stratum'] = pd.qcut(
        df['combined_length_score'], 
        q=n_bins, 
        labels=False, 
        duplicates='drop'
    )
    
    # Count samples in each stratum
    stratum_counts = df['stratum'].value_counts().sort_index()
    print(f"\nStratification stats ({len(stratum_counts)} strata):")
    for stratum, count in stratum_counts.items():
        print(f"  Stratum {stratum}: {count} samples")
    
    # Sample proportionally from each stratum
    test_indices = []
    remaining_per_stratum = stratum_counts.to_dict()
    
    # Calculate samples to draw from each stratum (proportionally)
    total_samples = len(df)
    samples_per_stratum = {}
    for stratum, count in remaining_per_stratum.items():
        proportion = count / total_samples
        samples_per_stratum[stratum] = max(1, int(n_test * proportion))
    
    # Ensure total count does not exceed 300
    total_allocated = sum(samples_per_stratum.values())
    if total_allocated > n_test:
        # Scale down proportionally
        scale = n_test / total_allocated
        samples_per_stratum = {k: max(1, int(v * scale)) for k, v in samples_per_stratum.items()}
    elif total_allocated < n_test:
        # Supplement from larger strata
        diff = n_test - total_allocated
        sorted_strata = sorted(remaining_per_stratum.items(), key=lambda x: x[1], reverse=True)
        for stratum, _ in sorted_strata:
            if diff > 0 and samples_per_stratum[stratum] < remaining_per_stratum[stratum]:
                samples_per_stratum[stratum] += 1
                diff -= 1
    
    # Randomly sample from each stratum (set random seed for reproducibility)
    np.random.seed(42)
    for stratum, n_samples in samples_per_stratum.items():
        stratum_data = df[df['stratum'] == stratum]
        if len(stratum_data) > 0:
            sampled = stratum_data.sample(n=min(n_samples, len(stratum_data)), random_state=42)
            test_indices.extend(sampled.index.tolist())
    
    # Ensure test set is exactly 300 samples (randomly adjust if not)
    np.random.seed(42)  # Set random seed again
    if len(test_indices) > n_test:
        test_indices = np.random.choice(test_indices, n_test, replace=False).tolist()
    elif len(test_indices) < n_test:
        remaining_indices = df[~df.index.isin(test_indices)].index.tolist()
        additional = np.random.choice(remaining_indices, n_test - len(test_indices), replace=False).tolist()
        test_indices.extend(additional)
    
    test_indices = set(test_indices)
    train_indices = set(df.index) - test_indices
    
    train_df = df.loc[list(train_indices)].copy()
    test_df = df.loc[list(test_indices)].copy()
    
    print(f"\nFinal split result:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    
    # Verify distribution similarity
    print("\nDistribution similarity check:")
    for col in columns:
        length_col = f'{col}_length'
        train_lengths = train_df[length_col]
        test_lengths = test_df[length_col]
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(train_lengths, test_lengths)
        
        print(f"\n{col} length distribution:")
        print(f"  Train - Mean: {train_lengths.mean():.2f}, Median: {train_lengths.median():.2f}, Std: {train_lengths.std():.2f}")
        print(f"  Test  - Mean: {test_lengths.mean():.2f}, Median: {test_lengths.median():.2f}, Std: {test_lengths.std():.2f}")
        print(f"  KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
        print(f"  {'Distributions are similar' if ks_pvalue > 0.05 else 'Distributions may differ'}")
    
    # Remove auxiliary columns
    cols_to_drop = [f'{col}_length' for col in columns] + ['combined_length_score', 'stratum']
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)
    
    return train_df, test_df


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Split train/test sets by length distribution")
    parser.add_argument(
        '--input',
        type=str,
        default='data/discharge_target.csv',
        help='Input discharge_target.csv path (default: data/discharge_target.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=300,
        help='Test set size (default: 300)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    print("=" * 80)
    print("Starting dataset split...")
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Test set size: {args.n_test}")
    print("=" * 80)
    
    train_df, test_df = stratified_sample_by_length(
        str(input_path),
        n_test=args.n_test,
        columns=['brief_hospital_course', 'discharge_instructions']
    )
    
    # Save train and test sets
    print("\nSaving datasets...")
    train_path = output_dir / 'discharge_target_train.csv'
    test_path = output_dir / 'discharge_target_test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n" + "=" * 80)
    print("Done!")
    print(f"Training set saved to: {train_path}")
    print(f"  Samples: {len(train_df)}")
    print(f"Test set saved to: {test_path}")
    print(f"  Samples: {len(test_df)}")
    print("=" * 80)




# python utils/split_dataset.py --input data/discharge_target.csv --output-dir data --n-test 300
