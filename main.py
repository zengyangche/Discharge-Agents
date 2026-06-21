"""
Main entry script: full multi-agent discharge summary generation system.

Pipeline:
  EHR → segmentation → 5 models each generate BHC+DI (N-shot / RAG)
       → multi-dimensional verification → best candidate → multi-model consensus → save

Few-shot experiment examples:
  # 0-shot, first 30 samples
  python main.py --csv-file discharge_target_test.csv --num-shots 0  --limit 30

  # 5/10/15/20-shot (static fixed examples)
  python main.py --csv-file discharge_target_test.csv --num-shots 5  --limit 30
  python main.py --csv-file discharge_target_test.csv --num-shots 10 --limit 30
  python main.py --csv-file discharge_target_test.csv --num-shots 15 --limit 30
  python main.py --csv-file discharge_target_test.csv --num-shots 20 --limit 30

  # RAG mode (top-K retrieve most similar examples)
  python main.py --csv-file discharge_target_test.csv --num-shots 5  --use-retrieval --limit 30
  python main.py --csv-file discharge_target_test.csv --num-shots 15 --use-retrieval --limit 30
"""
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from utils.config import OUTPUT_DIR, MODEL_ZOO
from core.multi_agent_framework import MultiAgentFramework
from utils.data_loader import DataLoader, load_sample_stay_ids
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


TEST_INPUTS_DIR = Path("outputs/test_inputs")


def _build_hadm_to_stay() -> dict:
    """
    Build hadm_id → stay_id mapping by scanning outputs/test_inputs/ filenames.
    Naming format: case_<stay_id>_<hadm_id>.txt
    Much faster than querying edstays.csv row by row.
    """
    mapping: dict = {}
    if not TEST_INPUTS_DIR.exists():
        return mapping
    for f in TEST_INPUTS_DIR.glob("case_*_*.txt"):
        parts = f.stem.split("_")   # ["case", stay_id, hadm_id]
        if len(parts) >= 3:
            mapping[parts[2]] = parts[1]  # hadm_id → stay_id
    return mapping


def _load_stay_ids_from_csv(csv_file: str, limit: Optional[int] = None) -> List[str]:
    """
    Load stay_id list from CSV (supports stay_id or hadm_id columns).
    If only hadm_id is present, convert via test_inputs directory scan (no DB lookup).
    limit: take the first N after sorting.
    """
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        csv_path = Path("data") / csv_path

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return []

    df_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

    if "stay_id" in df_cols:
        df = pd.read_csv(csv_path, usecols=["stay_id"], dtype={"stay_id": "Int64"})
        ids = sorted(df["stay_id"].dropna().unique().astype(str).tolist())
        logger.info(f"Read {len(ids)} stay_ids from CSV (stay_id column)")
        return ids[:limit] if limit else ids

    if "hadm_id" in df_cols:
        df = pd.read_csv(csv_path, usecols=["hadm_id"], dtype={"hadm_id": "Int64"})
        hadm_ids = df["hadm_id"].dropna().unique().astype(str).tolist()
        logger.info(
            f"CSV only has hadm_id column ({len(hadm_ids)} entries), "
            "converting to stay_id via test_inputs directory mapping…"
        )
        hadm_to_stay = _build_hadm_to_stay()
        stay_ids = []
        missing = 0
        for hid in hadm_ids:
            sid = hadm_to_stay.get(hid)
            if sid:
                stay_ids.append(sid)
            else:
                missing += 1
        stay_ids = sorted(set(stay_ids))
        if missing:
            logger.warning(
                f"{missing} hadm_id(s) have no matching file in test_inputs "
                f"(please run prepare_test_inputs.py first)"
            )
        logger.info(f"Conversion complete, {len(stay_ids)} stay_ids obtained")
        return stay_ids[:limit] if limit else stay_ids

    logger.error(f"CSV has no stay_id or hadm_id column. Available columns: {df_cols}")
    return []


def find_case_file(stay_id: str) -> Optional[Path]:
    """
    Find the EHR file for stay_id under outputs/test_inputs/.
    Naming format: case_<stay_id>_<hadm_id>.txt
    """
    if not TEST_INPUTS_DIR.exists():
        return None
    matches = list(TEST_INPUTS_DIR.glob(f"case_{stay_id}_*.txt"))
    # Exclude _final.txt
    matches = [f for f in matches if not f.name.endswith("_final.txt")]
    if matches:
        return matches[0]
    return None


def process_single_stay(
    stay_id: str,
    framework: MultiAgentFramework,
    save_output: bool = True,
    models: Optional[List[str]] = None,
) -> Dict:
    """
    Process a single stay_id.
    Locates the test_inputs file and calls MultiAgentFramework.process_case().
    """
    logger.info(f"\nProcessing stay_id: {stay_id}")
    logger.info("-" * 60)

    case_file = find_case_file(stay_id)
    if case_file is None:
        raise FileNotFoundError(
            f"No file found for stay_id={stay_id} in {TEST_INPUTS_DIR} "
            f"(case_{stay_id}_*.txt)"
        )

    result = framework.process_case(
        str(case_file),
        models=models,
        save_output=save_output,
        verbose=True,
    )
    return result


def print_result(result: Dict):
    """Print processing result summary"""
    logger.info("\n" + "=" * 60)
    logger.info("Final output summary")
    logger.info("=" * 60)

    summary = result.get("summary", {})
    consensus = summary.get("consensus_result", {})
    best_model = summary.get("best_model", "N/A")
    best_score = summary.get("best_score", 0)
    approved = consensus.get("approved", False)

    logger.info(f"Best model: {best_model}  Score: {best_score:.3f}")
    logger.info(f"Voting result: {'Approved' if approved else 'Not approved'}")

    final_bhc = consensus.get("final_bhc", "")
    final_di = consensus.get("final_di", "")
    if final_bhc:
        logger.info("\n[BHC preview (first 200 chars)]")
        logger.info(final_bhc[:200] + ("..." if len(final_bhc) > 200 else ""))
    if final_di:
        logger.info("\n[DI preview (first 200 chars)]")
        logger.info(final_di[:200] + ("..." if len(final_di) > 200 else ""))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-agent discharge summary generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process specified stay_id
  python main.py --stay-id 38112554
  
  # Process multiple stay_ids
  python main.py --stay-id 38112554 38112555 38112556
  
  # List available stay_ids
  python main.py --list-stays
  
  # Process using sample stay_ids
  python main.py --sample
  
  # View data statistics
  python main.py --stats
  
  # Read stay_ids from CSV and process
  python main.py --csv-file discharge_target_test.csv
  
  # Inspect CSV file structure
  python main.py --inspect-csv discharge_target_test.csv
        """
    )
    
    parser.add_argument(
        "--stay-id",
        nargs="+",
        type=str,
        help="stay_id(s) to process (one or more)"
    )
    parser.add_argument(
        "--list-stays",
        action="store_true",
        help="List all available stay_ids"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process using a sample stay_id from the dataset"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show data statistics"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o", "claude-3-5-sonnet-latest", "deepseek-v3", "gemini-2.5-flash", "grok-4"],
        help=f"Model list (default: all 5). Available: {', '.join(MODEL_ZOO.keys())}"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save outputs to files"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Read stay_id list from CSV (e.g. discharge_target_test.csv)"
    )
    parser.add_argument(
        "--stay-id-column",
        type=str,
        default=None,
        help="stay_id column name in CSV (if present)"
    )
    parser.add_argument(
        "--hadm-id-column",
        type=str,
        default=None,
        help="hadm_id column name in CSV (auto-converts to stay_id if needed)"
    )
    parser.add_argument(
        "--inspect-csv",
        type=str,
        help="Inspect CSV structure only (no processing)"
    )
    parser.add_argument(
        "--case-input",
        action="store_true",
        help="Show raw inputs (diagnosis/edstays/triage/radiology/discharge) and exit"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=3,
        choices=[0, 5, 10, 15, 20],
        metavar="{0,5,10,15,20}",
        help=(
            "Few-shot example count (default: 3, built-in fixed examples). "
            "Supported: 0, 5, 10, 15, 20"
        ),
    )
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help=(
            "Enable RAG mode: top-K vector retrieval instead of fixed examples. "
            "Requires --num-shots > 0 and labels under outputs/test_labels/."
        ),
    )
    parser.add_argument(
        "--retrieval-examples-dir",
        type=str,
        default="outputs/test_labels",
        help="RAG example pool directory (default: outputs/test_labels)",
    )
    parser.add_argument(
        "--retrieval-ehr-dir",
        type=str,
        default="outputs/test_inputs",
        help="RAG EHR embedding anchor directory (default: outputs/test_inputs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N samples after sorting (e.g. --limit 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Force output directory (default inferred from num-shots/use-retrieval)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Multi-Agent Discharge Summary Generation System")
    logger.info("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Handle different commands
    if args.inspect_csv:
        # Check CSV file structure
        from pathlib import Path
        csv_path = Path(args.inspect_csv)
        if not csv_path.is_absolute():
            csv_path = Path("data") / csv_path
        
        logger.info(f"\nInspecting CSV file structure: {csv_path}")
        structure = DataLoader.inspect_csv_structure(csv_path)
        
        if "error" in structure:
            logger.error(f"Error: {structure['error']}")
        else:
            logger.info(f"File path: {structure['path']}")
            logger.info(f"Columns: {', '.join(structure['columns'])}")
            if structure['total_rows']:
                logger.info(f"Total rows: {structure['total_rows']}")
            logger.info(f"\nFirst {structure['preview_rows']} rows:")
            for i, row in enumerate(structure['preview_data'][:3], 1):
                logger.info(f"\nRow {i}:")
                for key, value in row.items():
        # Only show non-null values, and limit display length
                    if pd.notna(value):
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        logger.info(f"  {key}: {value_str}")
        return
    
    if args.stats:
        # Display data statistics
        stats = data_loader.get_data_statistics()
        logger.info("\nData statistics:")
        logger.info(f"Files exist: {stats['files_exist']}")
        logger.info(f"Record counts: {stats['record_counts']}")
        logger.info(f"Available stay_id count: {stats['available_stay_ids']}")
        return
    
    if args.list_stays:
        # List available stay_ids
        logger.info("\nListing available stay_ids...")
        stay_ids = data_loader.list_available_stay_ids(limit=100, min_diagnosis_count=1)
        logger.info(f"\nFound {len(stay_ids)} stay_ids (showing first 100):")
        for i, sid in enumerate(stay_ids[:20], 1):  # show only first 20
            logger.info(f"  {i}. {sid}")
        if len(stay_ids) > 20:
            logger.info(f"  ... and {len(stay_ids) - 20} more")
        return
    
    # Determine stay_ids to process
    stay_ids_to_process = []
    
    if args.csv_file:
        logger.info(f"Reading stay_ids from CSV: {args.csv_file}")
        stay_ids_to_process = _load_stay_ids_from_csv(
            args.csv_file, limit=args.limit
        )
        if not stay_ids_to_process:
            logger.error(
                f"Unable to read stay_ids from {args.csv_file}. "
                "Please confirm the file exists and prepare_test_inputs.py has been run"
            )
            return
        logger.info(f"Total {len(stay_ids_to_process)} stay_ids")
        if len(stay_ids_to_process) > 10:
            logger.info(f"First 10: {stay_ids_to_process[:10]}")
    
    elif args.stay_id:
        stay_ids_to_process = args.stay_id
    elif args.sample:
        # Use sample stay_id
        sample_ids = load_sample_stay_ids(n=1, data_dir=None)
        if sample_ids:
            stay_ids_to_process = sample_ids
            logger.info(f"Using sample stay_id: {stay_ids_to_process[0]}")
        else:
            logger.error("No sample stay_id found, please check data files")
            return
    else:
        # Default behavior: use sample
        logger.info("No stay_id specified, using sample stay_id...")
        sample_ids = load_sample_stay_ids(n=1, data_dir=None)
        if sample_ids:
            stay_ids_to_process = sample_ids
            logger.info(f"Using sample stay_id: {stay_ids_to_process[0]}")
        else:
            logger.error("No sample stay_id found. Use --stay-id or --list-stays to view available IDs")
            return
    
    # Validate model names
    model_names = args.models
    invalid_models = [m for m in model_names if m not in MODEL_ZOO]
    if invalid_models:
        logger.error(f"Invalid model names: {invalid_models}")
        logger.error(f"Available models: {', '.join(MODEL_ZOO.keys())}")
        return

    # few-shot configuration log
    _shot_desc = f"{args.num_shots}-shot"
    if args.use_retrieval:
        _shot_desc += " [RAG]"
    logger.info(f"\nModels: {model_names}")
    logger.info(f"Few-shot config: {_shot_desc}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")

    try:
        # Initialize multi-agent framework (supports few-shot / RAG)
        framework = MultiAgentFramework(
            num_shots=args.num_shots,
            use_retrieval=args.use_retrieval,
            retrieval_examples_dir=args.retrieval_examples_dir,
            retrieval_ehr_dir=args.retrieval_ehr_dir,
            output_dir_override=args.output_dir,
        )

        # Process each stay_id
        for stay_id in stay_ids_to_process:
            try:
                if args.case_input:
                    # View raw input only and exit
                    case_file = find_case_file(stay_id)
                    if case_file is None:
                        logger.warning(f"No input file found for stay_id={stay_id}")
                        continue
                    ehr_text = case_file.read_text(encoding="utf-8")
                    logger.info("\n" + "=" * 60)
                    logger.info(f"Raw EHR file: {case_file.name}")
                    logger.info("First 500 chars preview:")
                    logger.info(ehr_text[:500])
                    continue

                result = process_single_stay(
                    stay_id,
                    framework,
                    save_output=not args.no_save,
                    models=model_names,
                )

                print_result(result)

                logger.info("\n" + "=" * 60)
                logger.info(f"stay_id {stay_id} processing complete!")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Failed to process stay_id {stay_id}: {str(e)}", exc_info=True)
                continue

    except Exception as e:
        logger.error(f"System initialization or processing failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

