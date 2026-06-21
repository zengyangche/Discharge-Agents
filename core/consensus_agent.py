"""
Consensus Agent
Summary and discussion agent: selects the best output and refines via multi-round discussion.
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
from utils.config import MODEL_ZOO

logger = logging.getLogger(__name__)


class ConsensusAgent:
    """
    Consensus agent: integrates outputs from multiple models via discussion and refinement.

    Flow:
    1. Score each candidate draft
    2. Select the best and runner-up drafts
    3. Run multi-LLM discussion to produce the final version
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ConsensusAgent")
    
    def finalize(
        self,
        candidates: Dict[str, Dict[str, Any]],  # {model_name: {output, verification_results}}
        verification_logs: Dict[str, Any],
        role: str,  # "bhc" or "di"
        discussion_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Finalize: select the best output and refine via discussion.

        Args:
            candidates: All candidate drafts and their verification results
            verification_logs: Verification logs
            role: Role ("bhc" or "di")
            discussion_models: Models used for discussion
            
        Returns:
            {
                "final_output": str,
                "selected_model": str,
                "discussion_log": [...],
                "metadata": {...}
            }
        """
        if discussion_models is None:
            discussion_models = list(MODEL_ZOO.keys())[:2]  # Use the first 2 models by default
        
        try:
            # 1. Score and rank
            ranked_candidates = self._rank_candidates(candidates)
            
            if not ranked_candidates:
                raise ValueError("No candidate drafts available")
            
            # 2. Select best and runner-up
            best_candidate = ranked_candidates[0]
            runner_up_candidates = ranked_candidates[1:3]  # Take top 3 (including best)
            
            self.logger.info(
                f"Best candidate selected: {best_candidate['model_name']} "
                f"(score: {best_candidate['score']:.3f})"
            )
            
            # 3. Conduct multi-round discussion
            final_output = self._discuss_and_refine(
                best_candidate,
                runner_up_candidates,
                verification_logs,
                role,
                discussion_models,
            )
            
            return {
                "final_output": final_output,
                "selected_model": best_candidate["model_name"],
                "discussion_log": [],  # TODO: Log discussion process
                "metadata": {
                    "best_score": best_candidate["score"],
                    "all_scores": {c["model_name"]: c["score"] for c in ranked_candidates},
                }
            }
        except Exception as e:
            self.logger.error(f"Finalization failed: {str(e)}")
            raise
    
    def _rank_candidates(
        self,
        candidates: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score and rank candidate drafts.

        Returns:
            Candidates sorted by score descending
        """
        ranked = []
        
        for model_name, candidate_data in candidates.items():
            if candidate_data is None or "output" not in candidate_data:
                continue
            
            # Get verification results
            # Note: the verification field is already bhc_verification or di_verification
            verification = candidate_data.get("verification", {})
            
            # Extract overall score directly (verification already holds the role-specific result)
            overall_score = verification.get("overall_score", 0.0)
            
            ranked.append({
                "model_name": model_name,
                "output": candidate_data["output"],
                "score": overall_score,
                "verification": verification,
            })
        
        # Sort by score in descending order
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked
    
    def _discuss_and_refine(
        self,
        best_candidate: Dict[str, Any],
        runner_up_candidates: List[Dict[str, Any]],
        verification_logs: Dict[str, Any],
        role: str,
        discussion_models: List[str],
    ) -> str:
        """
        Multi-round discussion and refinement.

        Uses multiple LLMs as an "expert committee" to discuss and optimize output.
        """
        # Build discussion prompt
        prompt = self._build_discussion_prompt(
            best_candidate,
            runner_up_candidates,
            verification_logs,
            role,
        )
        
        system_prompt = self._get_discussion_system_prompt(role)
        
        # Use the first discussion model to generate the final version
        # Can be extended to multi-round discussion
        try:
            from core.generator_agent import GeneratorAgent
            agent = GeneratorAgent(discussion_models[0], role, self.config)
            # Call LLM directly
            final_output = agent._call_llm(prompt, system_prompt)
            return final_output
        except Exception as e:
            self.logger.warning(f"Discussion generation failed, using best candidate: {str(e)}")
            return best_candidate["output"]
    
    def _build_discussion_prompt(
        self,
        best_candidate: Dict[str, Any],
        runner_up_candidates: List[Dict[str, Any]],
        verification_logs: Dict[str, Any],
        role: str,
    ) -> str:
        """Build discussion prompt"""
        parts = [
            f"As a senior attending physician, please review and refine the following {role.upper()} draft.",
            "",
            "=== Best Candidate Draft ===",
            f"Model: {best_candidate['model_name']}",
            f"Verification Score: {best_candidate['score']:.3f}",
            f"Content:\n{best_candidate['output']}",
            "",
        ]
        
        if runner_up_candidates:
            parts.append("=== Runner-up Candidate Drafts ===")
            for i, candidate in enumerate(runner_up_candidates, 1):
                parts.extend([
                    f"\nCandidate {i} (Model: {candidate['model_name']}, "
                    f"Score: {candidate['score']:.3f}):",
                    candidate['output'][:500] + "..." if len(candidate['output']) > 500
                    else candidate['output'],
                ])
            parts.append("")
        
        # Add errors from verification logs
        verification = verification_logs.get(f"{role}_verification", {})
        errors = []
        for verifier_name, result in verification.items():
            if isinstance(result, dict) and result.get("errors"):
                errors.extend(result["errors"])
        
        if errors:
            parts.extend([
                "=== Issues Found During Verification ===",
                "\n".join(f"- {error}" for error in errors[:10]),  # Show at most 10 errors
                "",
            ])
        
        parts.extend([
            "Based on the above information, please generate a refined final version:",
            "- Preserve sentences from the best candidate that are factually specific and not contradicted by alternatives",
            "- Use runner-up candidates only to fill GAPS or provide alternative phrasings for sections flagged as issues",
            "- Do NOT merge contradictory information across candidates — choose the more conservative or clinically safer version",
            "- Fix every issue listed in the verification section — leave none unaddressed",
            "- Write in English only",
        ])
        
        return "\n".join(parts)
    
    def _get_discussion_system_prompt(self, role: str) -> str:
        """Get discussion system prompt"""
        if role == "bhc":
            return """You are a senior attending physician responsible for reviewing and refining the Brief Hospital Course (BHC).

Your tasks:
1. Review the provided candidate drafts
2. Identify and fix all errors and issues
3. Integrate the strengths of multiple candidates
4. Generate an accurate, professional, and complete final BHC

Requirements:
- Maintain objectivity and accuracy
- Use professional yet clear medical terminology
- Organize content chronologically; use a numbered problem list for multiple diagnoses
- Ensure all key information is included
- Do not add information that does not appear in any of the candidate drafts
- Write in English only"""
        else:
            return """You are a senior attending physician responsible for reviewing and refining the Discharge Instructions (DI).

Your tasks:
1. Review the provided candidate drafts
2. Identify and fix all errors and issues
3. Integrate the strengths of multiple candidates
4. Generate a clear, complete, and patient-friendly final DI

Requirements:
- Use patient-friendly language; address the patient directly
- Ensure all necessary medical orders are included
- Be clear, specific, and actionable
- Fix all medical logic issues
- Do not add medications or dosages that do not appear in any of the candidate drafts
- Write in English only"""


class MultiAgentSystem:
    """
    Multi-agent system: integrates all pipeline components.
    """
    
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        if model_names is None:
            model_names = list(MODEL_ZOO.keys())
        
        self.model_names = model_names
        self.config = config or {}
        
        # Initialize logger first
        self.logger = logging.getLogger(f"{__name__}.MultiAgentSystem")
        
        # Initialize components
        from core.input_processor import InputProcessor
        from core.generator_agent import GeneratorOrchestrator
        from core.verification_engine import VerificationEngine
        from utils.knowledge_base import KnowledgeBase
        from pathlib import Path
        
        # Load knowledge base
        self.knowledge_base = None
        try:
            kb_dir = self.config.get("kb_dir", "outputs/knowledge_base")
            kb_path = Path(kb_dir)
            if kb_path.exists() and (kb_path / "bhc_centroid.npy").exists():
                self.knowledge_base = KnowledgeBase(kb_dir=kb_path)
                self.knowledge_base.load()
            self.logger.info(f"✓ Knowledge base loaded from {kb_dir}")
        else:
            self.logger.warning(f"Knowledge base directory not found or not built: {kb_dir}")
            self.logger.warning("Recommended: run python utils/knowledge_base.py to build the knowledge base")
        except Exception as e:
            self.logger.warning(f"Knowledge base failed to load: {str(e)}, falling back to degraded verification")
            self.logger.warning("Recommended: run python utils/knowledge_base.py to build the knowledge base")
        
        self.input_processor = InputProcessor(self.config)
        self.generator_orchestrator = GeneratorOrchestrator(model_names, self.config)
        # Pass knowledge base to verification engine
        self.verification_engine = VerificationEngine(self.config, knowledge_base=self.knowledge_base)
        self.consensus = ConsensusAgent(self.config)
    
    def process(self, stay_id: str) -> Dict[str, Any]:
        """
        Run the full processing pipeline.

        Args:
            stay_id: Hospital stay ID

        Returns:
            Complete processing result
        """
        self.logger.info(f"Starting processing of stay_id: {stay_id}")
        
        # 1. Input processing
        self.logger.info("Step 1: Input processing and decoupling...")
        input_data = self.input_processor.process(stay_id)
        
        # 2. Generation stage
        self.logger.info("Step 2: Multi-model candidate draft generation...")
        generation_results = self.generator_orchestrator.generate_all(
            input_data["shared_context"],
            input_data["bhc_context"],
            input_data["di_context"],
        )
        
        # 3. Verification stage
        self.logger.info("Step 3: Verifying all candidate drafts...")
        all_verification_results = {}
        
        for model_name in self.model_names:
            bhc_output = generation_results["bhc_outputs"].get(model_name)
            di_output = generation_results["di_outputs"].get(model_name)
            
            if bhc_output is None or di_output is None:
                continue
            
            verification_results = self.verification_engine.verify_all(
                bhc_output["output"],
                di_output["output"],
                input_data["shared_context"],
                input_data.get("raw_data"),
            )
            
            all_verification_results[model_name] = verification_results
        
        # 4. Consensus stage
        self.logger.info("Step 4: Consensus and finalization...")
        
        # Prepare candidate data
        bhc_candidates = {
            model_name: {
                "output": generation_results["bhc_outputs"][model_name]["output"],
                "role": "bhc",
                "verification": all_verification_results.get(model_name, {}).get(
                    "bhc_verification", {}
                ),
            }
            for model_name in self.model_names
            if generation_results["bhc_outputs"].get(model_name) is not None
        }
        
        di_candidates = {
            model_name: {
                "output": generation_results["di_outputs"][model_name]["output"],
                "role": "di",
                "verification": all_verification_results.get(model_name, {}).get(
                    "di_verification", {}
                ),
            }
            for model_name in self.model_names
            if generation_results["di_outputs"].get(model_name) is not None
        }
        
        # Finalize BHC and DI
        # Add verification results for each candidate
        for model_name in bhc_candidates:
            if model_name in all_verification_results:
                bhc_candidates[model_name]["verification"] = all_verification_results[model_name].get(
                    "bhc_verification", {}
                )
        
        for model_name in di_candidates:
            if model_name in all_verification_results:
                di_candidates[model_name]["verification"] = all_verification_results[model_name].get(
                    "di_verification", {}
                )
        
        bhc_final = self.consensus.finalize(
            bhc_candidates,
            all_verification_results.get(self.model_names[0], {}),  # Pass verification logs
            "bhc",
        )
        
        di_final = self.consensus.finalize(
            di_candidates,
            all_verification_results.get(self.model_names[0], {}),  # Pass verification logs
            "di",
        )
        
        return {
            "stay_id": stay_id,
            "input_data": input_data,
            "generation_results": generation_results,
            "verification_results": all_verification_results,
            "final_output": {
                "bhc": bhc_final,
                "di": di_final,
            },
        }
