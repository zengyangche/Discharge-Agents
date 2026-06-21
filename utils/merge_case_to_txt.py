"""
Merge all input data for a single case into one text file.
Simplifies the data loading workflow.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys

# Ensure DataLoader can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import DataLoader


def format_case_to_text(case_data: Dict[str, Any], include_metadata: bool = True) -> str:
    """
    Format case data as plain text (suitable for direct model input).

    Args:
        case_data: Data returned by load_case_inputs()
        include_metadata: Whether to include metadata (IDs, etc.)

    Returns:
        Formatted plain-text string (no decorative formatting)
    """
    lines = []
    
    # Metadata (ID info) - concise format
    if include_metadata and "ids" in case_data:
        ids = case_data["ids"]
        if ids.get('stay_id'):
            lines.append(f"Stay ID: {ids.get('stay_id')}")
        if ids.get('hadm_id'):
            lines.append(f"Hospital Admission ID: {ids.get('hadm_id')}")
        if ids.get('subject_id'):
            lines.append(f"Subject ID: {ids.get('subject_id')}")
        lines.append("")
    
    # Edstays information
    if case_data.get("edstays"):
        lines.append("Emergency Department Stay Information:")
        edstays = case_data["edstays"]
        for key, value in edstays.items():
            if value is not None and value != "":
                lines.append(f"{key}: {value}")
        lines.append("")
    
    # Diagnosis information
    if case_data.get("diagnosis"):
        lines.append("Diagnosis:")
        for diag in case_data["diagnosis"]:
            icd_title = diag.get('icd_title', 'N/A')
            icd_code = diag.get('icd_code', 'N/A')
            lines.append(f"- {icd_title} ({icd_code})")
        lines.append("")
    
    # Triage information
    if case_data.get("triage"):
        lines.append("Triage Information:")
        triage = case_data["triage"]
        for key, value in triage.items():
            if value is not None and value != "":
                lines.append(f"{key}: {value}")
        lines.append("")
    
    # Radiology information
    if case_data.get("radiology"):
        lines.append("Radiology Reports:")
        for rad in case_data["radiology"]:
            note_type = rad.get('note_type', 'N/A')
            note_id = rad.get('note_id', 'N/A')
            lines.append(f"Report Type: {note_type}, Note ID: {note_id}")
            if rad.get('charttime'):
                lines.append(f"Chart Time: {rad.get('charttime')}")
            if rad.get('storetime'):
                lines.append(f"Store Time: {rad.get('storetime')}")
            # Full text content
            if rad.get('text'):
                text = str(rad['text'])
                lines.append(f"Report Text: {text}")
            lines.append("")
    
    # Discharge information (BHC/DI labels removed)
    if case_data.get("discharge"):
        lines.append("Discharge Notes (BHC/DI sections removed):")
        for disc in case_data["discharge"]:
            note_type = disc.get('note_type', 'N/A')
            note_id = disc.get('note_id', 'N/A')
            lines.append(f"Note Type: {note_type}, Note ID: {note_id}")
            if disc.get('charttime'):
                lines.append(f"Chart Time: {disc.get('charttime')}")
            # Full text content (BHC/DI sections removed)
            if disc.get('text'):
                text = str(disc['text'])
                lines.append(f"Note Text: {text}")
            lines.append("")
    
    return "\n".join(lines)


def save_case_to_txt(
    case_data: Dict[str, Any],
    output_path: Union[str, Path],
    include_metadata: bool = True,
    include_json: bool = False
) -> Path:
    """
    Save case data to a text file.

    Args:
        case_data: Data returned by load_case_inputs()
        output_path: Output file path
        include_metadata: Whether to include metadata
        include_json: Whether to also append JSON at the end of the file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate text content
    text_content = format_case_to_text(case_data, include_metadata)
    
    # If needed, append JSON data (at end of file, separated by a simple marker)
    if include_json:
        text_content += "\n\n---JSON DATA---\n\n"
        text_content += json.dumps(case_data, ensure_ascii=False, indent=2)
    
    # Save file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    
    return output_path


def batch_merge_cases_to_txt(
    stay_ids: list,
    output_dir: Union[str, Path] = "outputs/merged_cases",
    data_dir: Optional[Path] = None,
    include_metadata: bool = True,
    include_json: bool = False
) -> Dict[str, Path]:
    """
    Batch-merge multiple cases into text files.

    Args:
        stay_ids: List of stay_ids
        output_dir: Output directory
        data_dir: Data directory (optional)
        include_metadata: Whether to include metadata
        include_json: Whether to include JSON data

    Returns:
        Dict mapping {stay_id: output_file_path}
    """
    loader = DataLoader(data_dir=data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for stay_id in stay_ids:
        try:
            # Load case data
            case_data = loader.load_case_inputs(
                stay_id=stay_id,
                drop_discharge_labels=True,
                strip_bhc_di_sections=True
            )
            
            # Generate filename
            hadm_id = case_data.get("ids", {}).get("hadm_id", "unknown")
            filename = f"case_{stay_id}_{hadm_id}.txt"
            output_path = output_dir / filename
            
            # Save file
            save_case_to_txt(
                case_data,
                output_path,
                include_metadata=include_metadata,
                include_json=include_json
            )
            
            results[str(stay_id)] = output_path
            print(f"✓ Saved: {stay_id} -> {output_path}")
            
        except Exception as e:
            print(f"✗ Failed to process {stay_id}: {str(e)}")
            continue
    
    return results


def load_case_from_txt(txt_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load case data from a text file (if it contains a JSON section).

    Args:
        txt_path: Text file path

    Returns:
        Case data dict if the file contains JSON; otherwise None
    """
    txt_path = Path(txt_path)
    
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")
    
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Locate JSON section
    json_marker = "---JSON DATA---"
    if json_marker in content:
        json_start = content.find(json_marker)
        json_content = content[json_start + len(json_marker):].strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            print("Warning: Unable to parse JSON section, returning None")
            return None
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge case input data into a single text file"
    )
    parser.add_argument(
        "--stay-id",
        type=str,
        help="Single stay_id or hadm_id"
    )
    parser.add_argument(
        "--hadm-id",
        type=str,
        help="hadm_id (ignored if --stay-id is provided)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Read ID list from a CSV file (batch processing)"
    )
    parser.add_argument(
        "--stay-id-column",
        type=str,
        default=None,
        help="stay_id column name in the CSV"
    )
    parser.add_argument(
        "--hadm-id-column",
        type=str,
        default=None,
        help="hadm_id column name in the CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/merged_cases",
        help="Output directory or file path"
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Include JSON data in the text file (for programmatic reads)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases in batch processing"
    )
    
    args = parser.parse_args()
    
    loader = DataLoader()
    
    # Determine IDs to process
    if args.csv_file:
        # Batch processing
        ids = DataLoader.load_stay_ids_from_csv(
            args.csv_file,
            stay_id_column=args.stay_id_column,
            hadm_id_column=args.hadm_id_column,
            limit=args.limit
        )
        print(f"Read {len(ids)} IDs from {args.csv_file}")
        
        output_dir = Path(args.output)
        results = batch_merge_cases_to_txt(
            ids,
            output_dir=output_dir,
            include_metadata=True,
            include_json=args.include_json
        )
        print(f"\nDone! Processed {len(results)} cases")
        
    elif args.stay_id or args.hadm_id:
        # Single case processing
        case_data = loader.load_case_inputs(
            stay_id=args.stay_id,
            hadm_id=args.hadm_id,
            drop_discharge_labels=True,
            strip_bhc_di_sections=True
        )
        
        # Determine output path
        if Path(args.output).suffix == ".txt":
            output_path = Path(args.output)
        else:
            stay_id = case_data.get("ids", {}).get("stay_id", "unknown")
            hadm_id = case_data.get("ids", {}).get("hadm_id", "unknown")
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"case_{stay_id}_{hadm_id}.txt"
        
        # Save
        save_case_to_txt(
            case_data,
            output_path,
            include_metadata=True,
            include_json=args.include_json
        )
        
        print(f"✓ Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
        
    else:
        parser.print_help()
        print("\nError: Must provide --stay-id, --hadm-id, or --csv-file")
