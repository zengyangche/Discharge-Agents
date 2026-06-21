"""
Multi-agent framework: coordinates segmentation, generation, and verification agents.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from agents.segmentation_agent import SegmentationAgent
from agents.generation_agent import GenerationAgent
from agents.verification_agent import VerificationAgent
from agents.summary_agent import SummaryAgent

logger = logging.getLogger(__name__)


class MultiAgentFramework:
    """
    Main multi-agent framework class.

    Workflow:
    1. Read EHR files from test_input
    2. Segment input into three contexts via segmentation agent (GPT-4O)
    3. Generate BHC and DI via generation agent (multiple models; N-shot / RAG)
    4. Multi-dimensional verification scoring
    5. Multi-model discussion and consensus refinement
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_shots: int = 3,
        use_retrieval: bool = False,
        retrieval_examples_dir: str = "outputs/test_labels",
        retrieval_ehr_dir: str = "outputs/test_inputs",
        retrieval_exclude_stems: Optional[List[str]] = None,
        output_dir_override: Optional[str] = None,
    ):
        """
        Args:
            config:                  Framework config (overrides MULTI_AGENT_CONFIG)
            num_shots:               Few-shot example count (0 = zero-shot; supports 0/5/10/15/20)
            use_retrieval:           If True, use top-K vector retrieval instead of fixed examples (RAG)
            retrieval_examples_dir:  RAG example pool directory (outputs/test_labels)
            retrieval_ehr_dir:       RAG EHR embedding anchor directory (outputs/test_inputs)
            retrieval_exclude_stems: Filename stems to exclude during retrieval (prevent test leakage)
            output_dir_override:     Force output directory (default inferred from num_shots/use_retrieval)
        """
        # Load default configuration from config file
        try:
            from utils.config import MULTI_AGENT_CONFIG
            default_config = MULTI_AGENT_CONFIG.copy()
        except ImportError:
            default_config = {}

        # Merge user config with default config
        if config:
            merged_config = default_config.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            self.config = merged_config
        else:
            self.config = default_config

        # ── few-shot parameters ───────────────────────────────────────────────
        self.num_shots = num_shots
        self.use_retrieval = use_retrieval and num_shots > 0

        # Initialize RAG retriever (if needed)
        retriever = None
        if self.use_retrieval:
            try:
                from utils.few_shot_retriever import FewShotRetriever
                logger.info(
                    f"[MultiAgentFramework] Initializing RAG retriever"
                    f" (example pool: {retrieval_examples_dir})..."
                )
                retriever = FewShotRetriever(
                    examples_dir=retrieval_examples_dir,
                    ehr_dir=retrieval_ehr_dir,
                    exclude_stems=retrieval_exclude_stems or [],
                )
                n_indexed = retriever.build_index()
                if n_indexed == 0:
                    logger.warning(
                        "[MultiAgentFramework] RAG index is empty, falling back to static few-shot mode"
                    )
                    retriever = None
                    self.use_retrieval = False
                else:
                    logger.info(
                        f"[MultiAgentFramework] RAG index ready, {n_indexed} examples indexed"
                    )
            except Exception as e:
                logger.error(
                    f"[MultiAgentFramework] RAG initialization failed, falling back to static few-shot: {e}"
                )
                retriever = None
                self.use_retrieval = False

        # Initialize agents
        segmentation_config = self.config.get("segmentation", {})
        generation_config = self.config.get("generation", {})
        verification_config = self.config.get("verification", {})

        self.segmentation_agent = SegmentationAgent(segmentation_config)
        self.generation_agent = GenerationAgent(
            generation_config,
            num_shots=num_shots,
            retriever=retriever,
        )
        
        # Initialize verification agent (requires knowledge base)
        knowledge_base = None
        try:
            from utils.knowledge_base import KnowledgeBase
            kb_dir = Path(self.config.get("knowledge_base_dir", "outputs/knowledge_base"))
            knowledge_base = KnowledgeBase(kb_dir=kb_dir)
            knowledge_base.load()
            logger.info("Knowledge base loaded successfully, verification agent initialized")
        except Exception as e:
            logger.warning(f"Knowledge base failed to load: {str(e)}, verification may be limited")
        
        self.verification_agent = VerificationAgent(verification_config, knowledge_base=knowledge_base)
        
        # Initialize summary agent
        summary_config = self.config.get("summary", {})
        self.summary_agent = SummaryAgent(summary_config)
        
        # Get generation model list from config
        self.default_models = self.config.get("generation", {}).get(
            "model_list",
            ["gpt-4o"]  # Default value
        )
        
        # Output directory: prefer output_dir_override, otherwise infer from shot config
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
        elif num_shots != 3 or self.use_retrieval:
            # few-shot experiments write to a dedicated directory, isolated from default results
            suffix = (
                f"{num_shots}shot_rag" if self.use_retrieval else f"{num_shots}shot"
            )
            self.output_dir = Path("outputs/multi_agent_few_shot") / suffix
        else:
            self.output_dir = Path(
                self.config.get("output_dir", "outputs/multi_agent_results")
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[MultiAgentFramework] Output directory: {self.output_dir}  "
            f"(num_shots={num_shots}, RAG={self.use_retrieval})"
        )
    
    def process_case(
        self,
        case_file_path: str,
        models: Optional[List[str]] = None,
        save_output: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single case.

        Args:
            case_file_path: EHR file path
            models: Model list; uses defaults if None
            save_output: Whether to save outputs
            verbose: Whether to print detailed progress

        Returns:
            Dict containing all pipeline results
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing case: {case_file_path}")
            print(f"{'='*80}")
        
        # Load EHR file
        ehr_text = self._load_ehr_file(case_file_path)
        if verbose:
            print(f"\n[1/5] EHR file loaded, length: {len(ehr_text)} chars")
        
        # Step 1: Segment input
        if verbose:
            print(f"\n[2/5] Segmenting input with segmentation agent...")
        segmentation_result = self.segmentation_agent.process(ehr_text)
        
        if verbose:
            print(f"  ✓ Shared context: {len(segmentation_result['shared_context'])} chars")
            print(f"  ✓ BHC-specific context: {len(segmentation_result['bhc_specific_context'])} chars")
            print(f"  ✓ DI-specific context: {len(segmentation_result['di_specific_context'])} chars")
            
            # Print segmentation result preview
            print(f"\nSegmentation preview:")
            print(f"{'-'*80}")
            print(f"[Shared Context] (first 300 chars):")
            print(segmentation_result['shared_context'][:300] + ("..." if len(segmentation_result['shared_context']) > 300 else ""))
            print(f"\n[BHC-Specific Context] (first 300 chars):")
            print(segmentation_result['bhc_specific_context'][:300] + ("..." if len(segmentation_result['bhc_specific_context']) > 300 else ""))
            print(f"\n[DI-Specific Context] (first 300 chars):")
            print(segmentation_result['di_specific_context'][:300] + ("..." if len(segmentation_result['di_specific_context']) > 300 else ""))
            print(f"{'-'*80}")
        
        # Save segmentation result to a separate file
        if save_output:
            self._save_segmentation_result(segmentation_result, case_file_path)
        
        # Step 2: Generate BHC and DI
        models_to_use = models or self.default_models
        if verbose:
            print(f"\n[3/5] Generating BHC and DI with {len(models_to_use)} models...")
            print(f"  Models: {', '.join(models_to_use)}")
        
        generation_result = self.generation_agent.batch_generate(
            shared_context=segmentation_result['shared_context'],
            bhc_specific_context=segmentation_result['bhc_specific_context'],
            di_specific_context=segmentation_result['di_specific_context'],
            model_names=models_to_use,
            show_progress=verbose  # Show progress bar if verbose is True
        )
        
        if verbose:
            print(f"\nGeneration results:")
            for model_name in models_to_use:
                bhc_len = len(generation_result['bhc_results'][model_name].get('content', ''))
                di_len = len(generation_result['di_results'][model_name].get('content', ''))
                print(f"  {model_name}:")
                print(f"    BHC: {bhc_len} chars")
                print(f"    DI: {di_len} chars")
        
        # Step 3: Verify generation results
        if verbose:
            print(f"\n[4/5] Verifying generation results with verification agent...")
        
        verification_result = self.verification_agent.process(
            bhc_results=generation_result.get('bhc_results', {}),
            di_results=generation_result.get('di_results', {}),
            ehr_text=ehr_text,
            shared_context=segmentation_result['shared_context'],
            bhc_specific_context=segmentation_result['bhc_specific_context'],
            di_specific_context=segmentation_result['di_specific_context']
        )
        
        if verbose:
            print(f"\nVerification results:")
            summary = verification_result.get('summary', {})
            model_scores = summary.get('model_scores', {})
            for model_name, score in model_scores.items():
                print(f"  {model_name}: {score:.3f}")
            best_model = summary.get('best_model')
            if best_model:
                print(f"  Best model: {best_model} (score: {model_scores.get(best_model, 0):.3f})")
        
        # Step 4: Summary agent (multi-model discussion)
        if verbose:
            print(f"\n[5/5] Running multi-model discussion with summary agent...")
        
        # Prepare generation results (extract BHC and DI content), only exclude obvious API error outputs
        generation_results_for_summary = {}
        error_keywords = ("[error", "error code:", "403 ", "api error", "rate limit",
                          "quota exceeded", "unauthorized", "permission denied",
                          "spending limit", "monthly limit", "[truncated]")
        for model_name in models_to_use:
            bhc_content = generation_result['bhc_results'][model_name].get('content', '')
            di_content  = generation_result['di_results'][model_name].get('content', '')
            bhc_lower = bhc_content.lower()
            di_lower  = di_content.lower()
            # API error keywords only checked at the beginning; truncation marker checked at the end
            bhc_is_error = (any(k in bhc_lower[:300] for k in error_keywords if k != "[truncated]")
                            or "[truncated]" in bhc_lower[-50:])
            di_is_error  = (any(k in di_lower[:300]  for k in error_keywords if k != "[truncated]")
                            or "[truncated]" in di_lower[-50:])
            if bhc_is_error or di_is_error:
                if verbose:
                    print(f"  [Filtered] Model {model_name} output contains API error, excluded from candidates")
                continue
            generation_results_for_summary[model_name] = {
                "bhc": bhc_content,
                "di": di_content
            }
        if not generation_results_for_summary:
            if verbose:
                print("  [Warning] All model outputs were filtered, falling back to longest raw output")
            best_fallback = max(
                models_to_use,
                key=lambda m: len(generation_result['bhc_results'][m].get('content', ''))
            )
            generation_results_for_summary[best_fallback] = {
                "bhc": generation_result['bhc_results'][best_fallback].get('content', ''),
                "di":  generation_result['di_results'][best_fallback].get('content', ''),
            }
        
        # Prepare verification results
        verification_results_for_summary = verification_result.get('model_results', {})
        verification_summary = verification_result.get('summary', {})
        
        summary_result = self.summary_agent.process(
            generation_results=generation_results_for_summary,
            verification_results=verification_results_for_summary,
            verification_summary=verification_summary,
            shared_context=segmentation_result['shared_context'],
            bhc_specific_context=segmentation_result['bhc_specific_context'],
            di_specific_context=segmentation_result['di_specific_context'],
        )
        
        if verbose:
            consensus = summary_result.get('consensus_result', {})
            approved = consensus.get('approved', False)
            approval_count = consensus.get('approval_count', 0)
            rejection_count = consensus.get('rejection_count', 0)
            voting_rounds = consensus.get('voting_rounds', 1)
            all_rounds = consensus.get('all_rounds', [])
            
            print(f"\nSummary results:")
            print(f"  Best model: {summary_result.get('best_model', 'N/A')}")
            print(f"  Voting rounds: {voting_rounds}")
            if all_rounds:
                print(f"  Per-round results:")
                for round_data in all_rounds:
                    round_num = round_data.get('round', 0)
                    round_approval = round_data.get('approval_count', 0)
                    round_rejection = round_data.get('rejection_count', 0)
                    print(f"    Round {round_num}: {round_approval} approve, {round_rejection} reject")
            print(f"  Final vote: {approval_count} approve, {rejection_count} reject")
            print(f"  Final status: {'PASSED' if approved else 'NOT PASSED'}")
            if consensus.get('modified', False):
                print(f"  Revision status: revised based on feedback")
        
        # Assemble results
        result = {
            "case_file": case_file_path,
            "segmentation": segmentation_result,
            "generation": generation_result,
            "verification": verification_result,
            "summary": summary_result,
            "models_used": models_to_use
        }
        
        # Save output
        if save_output:
            self._save_result(result, case_file_path)
            # Save verification logs
            case_name = Path(case_file_path).stem
            self.verification_agent.save_verification_logs(self.output_dir, case_name)
        
        return result
    
    def process_batch(
        self,
        case_files: List[str],
        models: Optional[List[str]] = None,
        save_output: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch-process multiple cases.

        Args:
            case_files: List of EHR file paths
            models: Model list to use
            save_output: Whether to save outputs
            verbose: Whether to print detailed progress

        Returns:
            List of per-case results
        """
        results = []
        total = len(case_files)
        
        if verbose:
            print(f"\nStarting batch processing of {total} cases...")
        
        for idx, case_file in enumerate(case_files, 1):
            if verbose:
                print(f"\n[{idx}/{total}] Processing: {Path(case_file).name}")
            
            try:
                result = self.process_case(
                    case_file, models, save_output, verbose=False
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {case_file}: {str(e)}")
                if verbose:
                    print(f"  ✗ Error: {str(e)}")
                results.append({
                    "case_file": case_file,
                    "error": str(e)
                })
        
        if verbose:
            print(f"\nBatch processing complete: {len(results)}/{total} succeeded")
        
        return results
    
    def _load_ehr_file(self, file_path: str) -> str:
        """Load EHR file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _save_segmentation_result(self, segmentation_result: Dict[str, Any], case_file_path: str):
        """Save segmentation results to a separate file"""
        case_name = Path(case_file_path).stem
        
        # Save segmentation result as JSON
        import json
        seg_json_file = self.output_dir / f"{case_name}_segmentation.json"
        with open(seg_json_file, 'w', encoding='utf-8') as f:
            json.dump(segmentation_result, f, ensure_ascii=False, indent=2)
        
        # Save segmentation result as text file
        seg_text_file = self.output_dir / f"{case_name}_segmentation.txt"
        with open(seg_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Case: {case_name}\n")
            f.write(f"File: {case_file_path}\n")
            f.write("="*80 + "\n\n")
            f.write("## Segmentation Result\n\n")
            
            f.write("### Shared Context\n")
            f.write("-"*80 + "\n")
            f.write(segmentation_result['shared_context'] + "\n\n")
            
            f.write("### BHC-Specific Context\n")
            f.write("-"*80 + "\n")
            f.write(segmentation_result['bhc_specific_context'] + "\n\n")
            
            f.write("### DI-Specific Context\n")
            f.write("-"*80 + "\n")
            f.write(segmentation_result['di_specific_context'] + "\n\n")
            
            # Save raw segmentation result if available
            if 'raw_segmentation' in segmentation_result:
                f.write("### Raw Segmentation Output\n")
                f.write("-"*80 + "\n")
                f.write(segmentation_result['raw_segmentation'] + "\n")
        
        logger.info(f"Segmentation result saved: {seg_json_file} and {seg_text_file}")
    
    def _save_result(self, result: Dict[str, Any], case_file_path: str):
        """Save results to file.

        Writes two files:
        1. Intermediate process file: segmentation, generation, verification, voting, etc.
        2. Final discharge summary file: final BHC and DI only (for metric evaluation)
        """
        case_name = Path(case_file_path).stem
        
        # Save complete results (JSON format)
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            return obj
        
        # Save complete JSON result (includes all intermediate steps)
        result_file = self.output_dir / f"{case_name}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            serializable_result = convert_to_serializable(result)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # Save intermediate process text file (includes all intermediate steps)
        intermediate_file = self.output_dir / f"{case_name}_intermediate.txt"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            f.write(f"Case: {case_name}\n")
            f.write(f"File: {case_file_path}\n")
            f.write("="*80 + "\n\n")
            
            # Segmentation result
            f.write("## Segmentation Result\n\n")
            f.write("### Shared Context:\n")
            f.write(result['segmentation']['shared_context'] + "\n\n")
            f.write("### BHC-Specific Context:\n")
            f.write(result['segmentation']['bhc_specific_context'] + "\n\n")
            f.write("### DI-Specific Context:\n")
            f.write(result['segmentation']['di_specific_context'] + "\n\n")
            
            # Generation results
            f.write("## Generation Results\n\n")
            for model_name in result['models_used']:
                f.write(f"### Model: {model_name}\n\n")
                
                f.write("#### Brief Hospital Course (BHC):\n")
                bhc_content = result['generation']['bhc_results'][model_name].get('content', '')
                f.write(bhc_content + "\n\n")
                
                f.write("#### Discharge Instructions (DI):\n")
                di_content = result['generation']['di_results'][model_name].get('content', '')
                f.write(di_content + "\n\n")
                
                # Verification results
                if 'verification' in result and 'model_results' in result['verification']:
                    verification = result['verification']['model_results'].get(model_name, {})
                    if verification:
                        f.write("#### Verification Scores:\n")
                        f.write(f"- Fact Verification: {verification.get('fact_verification', {}).get('score', 0):.3f}\n")
                        f.write(f"- Logic Verification: {verification.get('logic_verification', {}).get('score', 0):.3f}\n")
                        f.write(f"- Style Verification: {verification.get('style_verification', {}).get('score', 0):.3f}\n")
                        f.write(f"- Overall Score: {verification.get('overall_score', 0):.3f}\n")
                        
                        # Display improvement suggestions
                        improvement_log = verification.get('improvement_log', [])
                        if improvement_log:
                            f.write(f"\n#### Improvement Suggestions ({len(improvement_log)} items):\n")
                            for idx, item in enumerate(improvement_log[:5], 1):  # Show only the first 5
                                f.write(f"{idx}. [{item.get('type', 'unknown')}] {item.get('suggestion', '')}\n")
                            if len(improvement_log) > 5:
                                f.write(f"... and {len(improvement_log) - 5} more suggestions\n")
                        
                        # Display retrieved QA pairs (from logic verification)
                        logic_verification = verification.get('logic_verification', {})
                        if logic_verification:
                            sentence_pair_results = logic_verification.get('sentence_pair_results', [])
                            if sentence_pair_results:
                                f.write(f"\n#### Retrieved QA Pairs (logic verification):\n")
                                total_qa_pairs = 0
                                for pair_result in sentence_pair_results:
                                    qa_details = pair_result.get('retrieved_qa_details', [])
                                    if qa_details:
                                        total_qa_pairs += len(qa_details)
                                        f.write(f"\n  BHC Sentence {pair_result.get('bhc_sentence_index', '?')}:\n")
                                        f.write(f"    {pair_result.get('bhc_sentence', '')}\n")
                                        f.write(f"    Retrieved {len(qa_details)} QA pairs:\n")
                                        for qa_idx, qa in enumerate(qa_details[:3], 1):  # Show at most 3 per sentence
                                            f.write(f"      {qa_idx}. Q: {qa.get('question', '')[:150]}...\n")
                                            f.write(f"         A: {qa.get('answer', '')[:150]}...\n")
                                            f.write(f"         Similarity: {qa.get('similarity', 0):.3f}\n")
                                        if len(qa_details) > 3:
                                            f.write(f"      ... and {len(qa_details) - 3} more QA pairs\n")
                                f.write(f"\n  Total QA pairs retrieved: {total_qa_pairs}\n")
                        f.write("\n")
                
                f.write("-"*80 + "\n\n")
            
            # Verification summary
            if 'verification' in result and 'summary' in result['verification']:
                summary = result['verification']['summary']
                f.write("## Verification Summary\n\n")
                f.write(f"Best Model: {summary.get('best_model', 'N/A')}\n")
                f.write(f"Average Score: {summary.get('average_score', 0):.3f}\n")
                f.write("\n")
                f.write("Model Scores:\n")
                for model_name, score in summary.get('model_scores', {}).items():
                    f.write(f"  - {model_name}: {score:.3f}\n")
            
            # Final summary result
            if 'summary' in result:
                summary_result = result['summary']
                consensus = summary_result.get('consensus_result', {})
                f.write("\n## Final Summary Result\n\n")
                f.write(f"Best Model: {summary_result.get('best_model', 'N/A')}\n")
                f.write(f"Best Score: {summary_result.get('best_score', 0):.3f}\n")
                f.write(f"Discussion Models: {', '.join(summary_result.get('discussion_models', []))}\n")
                
                # Display multi-round voting results
                voting_rounds = consensus.get('voting_rounds', 1)
                all_rounds = consensus.get('all_rounds', [])
                
                if voting_rounds > 1 or all_rounds:
                    f.write(f"\nVoting Rounds: {voting_rounds}\n")
                    f.write(f"\n### Voting History:\n")
                    for round_data in all_rounds:
                        round_num = round_data.get('round', 0)
                        approval = round_data.get('approval_count', 0)
                        rejection = round_data.get('rejection_count', 0)
                        f.write(f"\n#### Round {round_num}:\n")
                        f.write(f"  Approval: {approval} votes\n")
                        f.write(f"  Rejection: {rejection} votes\n")
                        f.write(f"  Status: {'PASSED' if approval > rejection else 'FAILED'}\n")
                        f.write(f"  Votes:\n")
                        for model_name, vote in round_data.get('votes', {}).items():
                            f.write(f"    - {model_name}: {vote}\n")
                else:
                    # Single-round voting (backward compatible format)
                    f.write(f"\nVoting Results:\n")
                    votes = consensus.get('votes', {})
                    if isinstance(votes, dict):
                        # Check if votes are in list format (multi-round voting)
                        if votes and isinstance(list(votes.values())[0], list):
                            # Display votes from the last round
                            for model_name, vote_list in votes.items():
                                last_vote = vote_list[-1] if vote_list else "unknown"
                                f.write(f"  - {model_name}: {last_vote}\n")
                        else:
                            # Single-round voting format
                            for model_name, vote in votes.items():
                                f.write(f"  - {model_name}: {vote}\n")
                
                f.write(f"\nFinal Approval: {consensus.get('approval_count', 0)} votes\n")
                f.write(f"Final Rejection: {consensus.get('rejection_count', 0)} votes\n")
                f.write(f"Final Status: {'APPROVED' if consensus.get('approved', False) else 'REJECTED'}\n")
                
                if consensus.get('modified', False):
                    f.write(f"Modified: Yes (after {voting_rounds} round(s))\n")
                
                if consensus.get('approved', False) or consensus.get('final_bhc') or consensus.get('final_di'):
                    f.write("\n### Final Output:\n\n")
                    f.write("#### Brief Hospital Course (BHC):\n")
                    f.write(consensus.get('final_bhc', '') + "\n\n")
                    f.write("#### Discharge Instructions (DI):\n")
                    f.write(consensus.get('final_di', '') + "\n\n")
                    
                    modification_log = consensus.get('modification_log', [])
                    if modification_log:
                        f.write("#### Modification Log:\n")
                        for log_item in modification_log:
                            if log_item.get('type') == 'improvement_suggestions':
                                f.write(f"- Addressed {log_item.get('count', 0)} improvement suggestions\n")
                            elif log_item.get('type') == 'rejection_reasons':
                                f.write(f"- Addressed rejection reasons from peer review\n")
                            elif log_item.get('type') == 'error':
                                f.write(f"- Error: {log_item.get('message', '')}\n")
                else:
                    f.write("\n### Final Output: REJECTED (No output generated)\n")
        
        # Save final discharge summary file (only contains final BHC and DI, for metric computation)
        final_file = self.output_dir / f"{case_name}_final.txt"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(f"Case: {case_name}\n")
            f.write(f"File: {case_file_path}\n")
            f.write("="*80 + "\n\n")
            
            # Only save the final BHC and DI
            if 'summary' in result:
                summary_result = result['summary']
                consensus = summary_result.get('consensus_result', {})
                
                if consensus.get('approved', False) or consensus.get('final_bhc') or consensus.get('final_di'):
                    f.write("## Brief Hospital Course (BHC)\n\n")
                    f.write(consensus.get('final_bhc', '') + "\n\n")
                    f.write("="*80 + "\n\n")
                    f.write("## Discharge Instructions (DI)\n\n")
                    f.write(consensus.get('final_di', '') + "\n\n")
                else:
                    f.write("## Final Output: REJECTED (No output generated)\n")
            else:
                f.write("## Final Output: Not available\n")
        
        logger.info(f"Results saved:")
        logger.info(f"  - Intermediate file: {intermediate_file}")
        logger.info(f"  - Final discharge summary: {final_file}")
        logger.info(f"  - Complete JSON result: {result_file}")
