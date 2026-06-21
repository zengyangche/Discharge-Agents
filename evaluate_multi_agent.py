"""
Evaluate multi-agent system generation outputs.

Reads *_final.txt files from outputs/multi_agent_results (or a custom dir).
Loads reference labels from outputs/test_labels or data/discharge_target_test.csv.
Computes metrics via the scoring module.

# 15-shot RAG
python evaluate_multi_agent.py --generated-dir outputs/multi_agent_few_shot/15shot_rag --output-dir outputs/evaluation/multi_agent_15shot_rag
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentEvaluator:
    """Multi-agent system evaluator"""
    
    def __init__(
        self,
        generated_dir: str = "outputs/multi_agent_results",
        reference_file: str = "data/discharge_target_test.csv",
        output_dir: str = "outputs/evaluation"
    ):
        """
        Initialize the evaluator.

        Args:
            generated_dir: Directory with generated *_final.txt files
            reference_file: Reference labels CSV file
            output_dir: Evaluation output directory
        """
        self.generated_dir = Path(generated_dir)
        self.reference_file = Path(reference_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generated results directory: {self.generated_dir}")
        logger.info(f"Reference labels file: {self.reference_file}")
        logger.info(f"Evaluation output directory: {self.output_dir}")
    
    def parse_final_txt(self, file_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a *_final.txt file and extract hadm_id, BHC, and DI.

        Args:
            file_path: Path to the final.txt file

        Returns:
            (hadm_id, bhc_content, di_content)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract hadm_id from filename (format: case_<stay_id>_<hadm_id>_final.txt)
            filename = file_path.stem  # strip .txt extension
            parts = filename.split('_')
            if len(parts) >= 3:
                hadm_id = parts[2]  # Extract hadm_id
            else:
                logger.warning(f"Unable to extract hadm_id from filename: {file_path}")
                return None, None, None
            
            # Extract BHC (between ## Brief Hospital Course (BHC) and first ==)
            bhc_pattern = r'## Brief Hospital Course \(BHC\)\s*\n(.*?)(?=\n={50,}|\n## Discharge Instructions|\Z)'
            bhc_match = re.search(bhc_pattern, content, re.DOTALL)
            bhc_content = bhc_match.group(1).strip() if bhc_match else ""
            
            # Extract DI (after ## Discharge Instructions (DI))
            di_pattern = r'## Discharge Instructions \(DI\)\s*\n(.*?)(?=\Z)'
            di_match = re.search(di_pattern, content, re.DOTALL)
            di_content = di_match.group(1).strip() if di_match else ""
            
            # Clean content: remove extra tags and prefixes
            bhc_content = self._clean_content(bhc_content)
            di_content = self._clean_content(di_content)
            
            return hadm_id, bhc_content, di_content
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {str(e)}")
            return None, None, None
    
    def _clean_content(self, content: str) -> str:
        """
        Clean content: remove extra labels and prefixes.

        Args:
            content: Raw content

        Returns:
            Cleaned content
        """
        # Remove "Brief Hospital Course:" prefix
        content = re.sub(r'^Brief Hospital Course:\s*\n', '', content, flags=re.MULTILINE)
        
        # Remove "Discharge Instructions:" prefix
        content = re.sub(r'^Discharge Instructions:\s*\n', '', content, flags=re.MULTILINE)
        
        # Remove extra blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def load_generated_results(self) -> pd.DataFrame:
        """
        Load all generated results.

        Returns:
            DataFrame with hadm_id, brief_hospital_course, discharge_instructions columns
        """
        logger.info("Loading generated results...")
        
        # Find all _final.txt files
        final_files = list(self.generated_dir.glob("*_final.txt"))
        logger.info(f"Found {len(final_files)} final files")
        
        if len(final_files) == 0:
            logger.error(f"No _final.txt files found in {self.generated_dir}")
            raise FileNotFoundError(f"No _final.txt files found in {self.generated_dir}")
        
        results = []
        for file_path in tqdm(final_files, desc="Parsing generated results"):
            hadm_id, bhc, di = self.parse_final_txt(file_path)
            if hadm_id:
                results.append({
                    "hadm_id": int(hadm_id),
                    "brief_hospital_course": bhc if bhc else "",
                    "discharge_instructions": di if di else ""
                })
        
        df = pd.DataFrame(results)
        logger.info(f"Successfully loaded {len(df)} samples")
        
        # Check for empty content
        empty_bhc = (df["brief_hospital_course"] == "").sum()
        empty_di = (df["discharge_instructions"] == "").sum()
        if empty_bhc > 0:
            logger.warning(f"Found {empty_bhc} samples with empty BHC")
        if empty_di > 0:
            logger.warning(f"Found {empty_di} samples with empty DI")
        
        return df
    
    def load_reference_labels(self) -> pd.DataFrame:
        """
        Load reference labels.

        Returns:
            DataFrame with hadm_id, brief_hospital_course, discharge_instructions columns
        """
        logger.info("Loading reference labels...")
        
        if not self.reference_file.exists():
            raise FileNotFoundError(f"Reference labels file not found: {self.reference_file}")
        
        # Read CSV file
        df = pd.read_csv(self.reference_file, keep_default_na=False)
        logger.info(f"Successfully loaded {len(df)} reference samples")
        
        # Ensure required columns exist
        required_columns = ["hadm_id", "brief_hospital_course", "discharge_instructions"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Reference file missing required columns: {missing_columns}")
        
        # Convert hadm_id to int
        df["hadm_id"] = df["hadm_id"].astype(int)
        
        # Ensure content is string
        df["brief_hospital_course"] = df["brief_hospital_course"].astype(str)
        df["discharge_instructions"] = df["discharge_instructions"].astype(str)
        
        return df
    
    def align_data(
        self,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align generated results with reference labels (by hadm_id).

        Args:
            generated_df: Generated results DataFrame
            reference_df: Reference labels DataFrame

        Returns:
            (aligned_generated, aligned_reference)
        """
        logger.info("Aligning data...")
        
        # Find common hadm_ids
        common_hadm_ids = set(generated_df["hadm_id"]) & set(reference_df["hadm_id"])
        logger.info(f"Found {len(common_hadm_ids)} common hadm_ids")
        
        if len(common_hadm_ids) == 0:
            logger.error("No common hadm_ids between generated results and reference labels")
            raise ValueError("No common hadm_ids between generated and reference data")
        
        # Filter and sort
        generated_aligned = generated_df[generated_df["hadm_id"].isin(common_hadm_ids)].copy()
        reference_aligned = reference_df[reference_df["hadm_id"].isin(common_hadm_ids)].copy()
        
        generated_aligned = generated_aligned.sort_values("hadm_id").reset_index(drop=True)
        reference_aligned = reference_aligned.sort_values("hadm_id").reset_index(drop=True)
        
        # Check for missing hadm_ids
        missing_in_generated = set(reference_df["hadm_id"]) - set(generated_df["hadm_id"])
        missing_in_reference = set(generated_df["hadm_id"]) - set(reference_df["hadm_id"])
        
        if missing_in_generated:
            logger.warning(f"{len(missing_in_generated)} hadm_ids in reference labels are missing from generated results")
        if missing_in_reference:
            logger.warning(f"{len(missing_in_reference)} hadm_ids in generated results are missing from reference labels")
        
        return generated_aligned, reference_aligned
    
    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess text: convert to single-line strings (remove newlines).

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        logger.info("Preprocessing text...")
        
        df = df.copy()
        
        # Convert to single-line strings
        df["brief_hospital_course"] = df["brief_hospital_course"].str.replace("\n", " ")
        df["discharge_instructions"] = df["discharge_instructions"].str.replace("\n", " ")
        
        # Remove extra whitespace
        df["brief_hospital_course"] = df["brief_hospital_course"].str.replace(r'\s+', ' ', regex=True)
        df["discharge_instructions"] = df["discharge_instructions"].str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def calculate_basic_metrics(
        self,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame
    ) -> Dict:
        """
        Compute basic metrics (no external scoring libraries required).

        Args:
            generated_df: Generated results DataFrame
            reference_df: Reference labels DataFrame

        Returns:
            Metrics dict
        """
        logger.info("Calculating basic metrics...")
        
        metrics = {
            "sample_count": len(generated_df),
            "bhc_metrics": {},
            "di_metrics": {}
        }
        
        # Calculate average lengths
        for field, label in [("brief_hospital_course", "bhc"), ("discharge_instructions", "di")]:
            gen_lengths = generated_df[field].str.split().str.len()
            ref_lengths = reference_df[field].str.split().str.len()
            
            metrics[f"{label}_metrics"]["avg_generated_length"] = float(gen_lengths.mean())
            metrics[f"{label}_metrics"]["avg_reference_length"] = float(ref_lengths.mean())
            metrics[f"{label}_metrics"]["length_ratio"] = float(gen_lengths.mean() / ref_lengths.mean())
        
        return metrics
    
    def calculate_scoring_metrics(
        self,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        metrics: List[str] = ["bleu", "rouge", "bertscore", "meteor"]
    ) -> Dict:
        """
        Compute evaluation metrics using the scoring module.

        Args:
            generated_df: Generated results DataFrame
            reference_df: Reference labels DataFrame
            metrics: Metrics to compute (bleu, rouge, bertscore, meteor, align)

        Returns:
            Scoring results dict
        """
        logger.info(f"Calculating scoring metrics: {metrics}...")
        
        # Filter out medcon (requires UMLSScorer)
        if "medcon" in metrics:
            logger.warning("medcon metric requires UMLSScorer configuration, skipping automatically")
            metrics = [m for m in metrics if m != "medcon"]
        
        if not metrics:
            logger.warning("No valid scoring metrics available")
            return {"error": "no valid metrics specified"}
        
        try:
            # Import scoring module
            import sys
            scoring_dir = Path(__file__).parent / "scoring"
            sys.path.insert(0, str(scoring_dir))
            
            from scoring import calculate_scores, compute_overall_score
            
            # Calculate scores
            scores = calculate_scores(generated_df, reference_df, metrics)
            
            # Calculate overall score
            leaderboard = compute_overall_score(scores)
            
            return {
                "detailed_scores": scores,
                "leaderboard": leaderboard
            }
            
        except ImportError as e:
            logger.warning(f"Unable to import scoring module: {str(e)}")
            logger.warning("Skipping advanced scoring metrics")
            return {"error": f"scoring module not available: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to calculate scoring metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def save_results(
        self,
        basic_metrics: Dict,
        scoring_metrics: Dict,
        generated_df: pd.DataFrame,
        reference_df: pd.DataFrame
    ):
        """
        Save evaluation results.

        Args:
            basic_metrics: Basic metrics
            scoring_metrics: Scoring metrics
            generated_df: Generated results DataFrame
            reference_df: Reference labels DataFrame
        """
        logger.info("Saving evaluation results...")
        
        # Save summary metrics
        summary = {
            "basic_metrics": basic_metrics,
            "scoring_metrics": scoring_metrics
        }
        
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary metrics saved: {summary_file}")
        
        # Save aligned data (for subsequent analysis)
        aligned_file = self.output_dir / "aligned_data.csv"
        aligned_df = generated_df.copy()
        aligned_df["ref_brief_hospital_course"] = reference_df["brief_hospital_course"]
        aligned_df["ref_discharge_instructions"] = reference_df["discharge_instructions"]
        aligned_df.to_csv(aligned_file, index=False, encoding='utf-8')
        logger.info(f"Aligned data saved: {aligned_file}")
        
        # Generate readable report
        self._generate_report(basic_metrics, scoring_metrics)
    
    def _generate_report(self, basic_metrics: Dict, scoring_metrics: Dict):
        """
        Generate a human-readable evaluation report.

        Args:
            basic_metrics: Basic metrics
            scoring_metrics: Scoring metrics
        """
        report_file = self.output_dir / "evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-Agent System Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metrics
            f.write("## Basic Metrics\n\n")
            f.write(f"Sample count: {basic_metrics['sample_count']}\n\n")
            
            f.write("### Brief Hospital Course (BHC)\n")
            bhc = basic_metrics['bhc_metrics']
            f.write(f"- Avg generated length: {bhc['avg_generated_length']:.1f} words\n")
            f.write(f"- Avg reference length: {bhc['avg_reference_length']:.1f} words\n")
            f.write(f"- Length ratio: {bhc['length_ratio']:.3f}\n\n")
            
            f.write("### Discharge Instructions (DI)\n")
            di = basic_metrics['di_metrics']
            f.write(f"- Avg generated length: {di['avg_generated_length']:.1f} words\n")
            f.write(f"- Avg reference length: {di['avg_reference_length']:.1f} words\n")
            f.write(f"- Length ratio: {di['length_ratio']:.3f}\n\n")
            
            # Scoring metrics
            if "leaderboard" in scoring_metrics:
                f.write("## Scoring Metrics\n\n")
                leaderboard = scoring_metrics["leaderboard"]
                
                for metric, score in sorted(leaderboard.items()):
                    f.write(f"- {metric.upper()}: {score:.4f}\n")
                
                f.write(f"\n**Overall Score: {leaderboard.get('overall', 0):.4f}**\n\n")
            elif "error" in scoring_metrics:
                f.write("## Scoring Metrics\n\n")
                f.write(f"Error: {scoring_metrics['error']}\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"Evaluation report saved: {report_file}")
    
    def run(self, metrics: List[str] = None):
        """
        Run the full evaluation pipeline.

        Args:
            metrics: Metrics to compute (default: bleu, rouge, bertscore, meteor, align)
        """
        # Default to all available metrics
        if metrics is None:
            metrics = ["bleu", "rouge", "bertscore", "meteor", "align"]
        logger.info("=" * 80)
        logger.info("Starting multi-agent system evaluation")
        logger.info("=" * 80)
        
        try:
            # 1. Load data
            generated_df = self.load_generated_results()
            reference_df = self.load_reference_labels()
            
            # 2. Align data
            generated_aligned, reference_aligned = self.align_data(generated_df, reference_df)
            
            # 3. Preprocess text
            generated_preprocessed = self.preprocess_text(generated_aligned)
            reference_preprocessed = self.preprocess_text(reference_aligned)
            
            # 4. Calculate basic metrics
            basic_metrics = self.calculate_basic_metrics(
                generated_preprocessed, reference_preprocessed
            )
            
            # 5. Calculate scoring metrics
            scoring_metrics = self.calculate_scoring_metrics(
                generated_preprocessed, reference_preprocessed, metrics
            )
            
            # 6. Save results
            self.save_results(
                basic_metrics, scoring_metrics,
                generated_preprocessed, reference_preprocessed
            )
            
            logger.info("=" * 80)
            logger.info("Evaluation complete!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 80)
            
            # Print summary
            print("\n" + "=" * 80)
            print("Evaluation Summary")
            print("=" * 80)
            print(f"Sample count: {basic_metrics['sample_count']}")
            print(f"\nBHC length ratio: {basic_metrics['bhc_metrics']['length_ratio']:.3f}")
            print(f"DI length ratio: {basic_metrics['di_metrics']['length_ratio']:.3f}")
            
            if "leaderboard" in scoring_metrics:
                print("\nScoring results:")
                for metric, score in sorted(scoring_metrics["leaderboard"].items()):
                    print(f"  {metric.upper()}: {score:.4f}")
            
            print("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent system generation outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Evaluate with defaults (all available metrics)
  python evaluate_multi_agent.py
  
  # Specify custom paths
  python evaluate_multi_agent.py --generated-dir outputs/multi_agent_results --reference-file data/discharge_target_test.csv
  
  # Fast metrics only (BLEU and ROUGE)
  python evaluate_multi_agent.py --metrics bleu rouge
  
  # BLEU only
  python evaluate_multi_agent.py --metrics bleu
  
  # Specify output directory
  python evaluate_multi_agent.py --output-dir outputs/my_evaluation
        """
    )
    
    parser.add_argument(
        "--generated-dir",
        type=str,
        default="outputs/multi_agent_results",
        help="Generated results directory (contains *_final.txt files)"
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default="data/discharge_target_test.csv",
        help="Reference labels file (CSV format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Evaluation output directory"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["bleu", "rouge", "bertscore", "meteor", "align"],
        help="Metrics to compute (default: all). Note: bertscore/meteor download models on first run; align needs AlignScore"
    )
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = MultiAgentEvaluator(
        generated_dir=args.generated_dir,
        reference_file=args.reference_file,
        output_dir=args.output_dir
    )
    
    evaluator.run(metrics=args.metrics)


if __name__ == "__main__":
    main()
