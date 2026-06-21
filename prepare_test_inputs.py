"""
Prepare test-set input files for zero-shot inference.
Reads IDs from discharge_target_test.csv and writes merged input text files.
"""
import sys
from pathlib import Path

# Ensure imports are possible
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.merge_case_to_txt import batch_merge_cases_to_txt
from utils.data_loader import DataLoader
import argparse


def prepare_test_inputs(
    test_file: str = "data/discharge_target_test.csv",
    output_dir: str = "outputs/test_inputs",
    include_json: bool = False,
    limit: int = None,
    data_dir: str = "data"
):
    """
    Prepare input files for the test set.

    Args:
        test_file: Test set path (discharge_target_test.csv)
        output_dir: Output directory
        include_json: Whether to append JSON data
        limit: Max samples to process (None = all)
        data_dir: Data directory for raw source files
    """
    test_file_path = Path(test_file)
    data_dir_path = Path(data_dir)
    
    # Check test set file
    if not test_file_path.exists():
        print(f"Error: Test set file not found: {test_file_path}")
        return
    
    print(f"=" * 80)
    print(f"Preparing test set input files")
    print(f"=" * 80)
    print(f"Test set file: {test_file_path}")
    print(f"Data directory: {data_dir_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create DataLoader using the merged data directory
    loader = DataLoader(data_dir=data_dir_path)
    
    # Read IDs from discharge_target_test.csv
    print("Reading test set IDs...")
    try:
        # Try reading hadm_id column first (preferred)
        ids = DataLoader.load_stay_ids_from_csv(
            test_file_path,
            hadm_id_column="hadm_id",
            data_dir=data_dir_path,
            limit=limit
        )
        
        if not ids:
            # Try stay_id column
            ids = DataLoader.load_stay_ids_from_csv(
                test_file_path,
                stay_id_column="stay_id",
                data_dir=data_dir_path,
                limit=limit
            )
        
        if not ids:
            print("Error: Unable to read IDs from discharge_target_test.csv")
            print("Please check that the file contains hadm_id or stay_id columns")
            return
        
        print(f"Found {len(ids)} test samples")
        print()
        
        # Batch-generate merged text files
        print("Generating merged input files...")
        results = batch_merge_cases_to_txt(
            ids,
            output_dir=output_dir,
            data_dir=data_dir_path,  # Use merged data directory
            include_metadata=True,
            include_json=include_json
        )
        
        print()
        print("=" * 80)
        print(f"Done! Successfully generated {len(results)} input files")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        
        # List generated files
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("case_*.txt"))
            print(f"\nGenerated files (first 10):")
            for f in files[:10]:
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.2f} KB)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test-set input files for zero-shot inference (from discharge_target_test.csv)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/discharge_target_test.csv",
        help="Test set file path (default: data/discharge_target_test.csv)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory for raw files (default: data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/test_inputs",
        help="Output directory (default: outputs/test_inputs)"
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Include JSON data in the text file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing; default: all)"
    )
    
    args = parser.parse_args()
    
    prepare_test_inputs(
        test_file=args.test_file,
        data_dir=args.data_dir,
        output_dir=args.output,
        include_json=args.include_json,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

# python prepare_test_inputs.py --test-file data/discharge_target_test.csv --output outputs/test_inputs