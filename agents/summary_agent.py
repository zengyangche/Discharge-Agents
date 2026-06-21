"""
Summary Agent: selects the best output based on verification results and reaches consensus via multi-model discussion
"""
from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
import logging
from collections import Counter

logger = logging.getLogger(__name__)

import re


def _strip_modification_metadata(text: str, doc_type: str = "bhc") -> str:
    """
    Strip metadata annotations from LLM revision/generation responses.
    Covers common patterns:
      - "Revised BHC:" / "Revised Brief Hospital Course (BHC):" leading headers
      - "Brief Hospital Course start:" / "Brief Hospital Course end." wrapper headers
      - "Key Changes Made:" / "Key Revisions:" / "Key Synthesis Notes:" trailing explanation blocks
      - Content after "---" dividers (e.g. reasoning notes appended by grok)
    """
    if not text:
        return text

    # ── Remove leading metadata header lines ──────────────────────────────────────────────
    opening_patterns = [
        # **Revised Brief Hospital Course (BHC):** / Revised DI:
        r'^\s*\*{0,2}Revised\s+(?:Brief\s+Hospital\s+Course|Discharge\s+Instructions?)'
        r'(?:\s*\([^)]*\))?\s*:?\*{0,2}\s*\n?',
        # Brief Hospital Course start: / Brief Hospital Course:
        r'^\s*\*{0,2}Brief\s+Hospital\s+Course\s*(?:start\s*)?:?\*{0,2}\s*\n?',
        # Discharge Instructions: (as a standalone header)
        r'^\s*\*{0,2}Discharge\s+Instructions?\s*:?\*{0,2}\s*\n?',
    ]
    for pat in opening_patterns:
        text = re.sub(pat, '', text, count=1, flags=re.IGNORECASE)

    # ── Truncate trailing explanation blocks (keep clinical body, remove model reasoning notes) ───────────────
    cutoff_pattern = re.compile(
        r'\n\s*(?:'
        r'---+'                                        # --- section divider
        r'|\*{0,2}Key\s+(?:Changes?\s+Made|Revisions?|Synthesis\s+Notes?)\*{0,2}'
        r'|Brief\s+Hospital\s+Course\s+end'           # "Brief Hospital Course end."
        r'|Discharge\s+Instructions?\s+end'
        r')',
        re.IGNORECASE
    )
    m = cutoff_pattern.search(text)
    if m:
        text = text[:m.start()]

    return text.strip()


class SummaryAgent(BaseAgent):
    """
    Summary Agent: pick highest-scoring output and reach consensus via multi-model discussion
    
    Workflow:
    1. Select the highest-scoring model output from verification results as the primary candidate
    2. Collect runner-up results and verification logs
    3. Build discussion prompt and have multiple models vote
    4. Majority approval mechanism produces the final output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SummaryAgent", config)
        self.llm_client = LLMClient()
        
        # Read discussion model list from config (defaults to the generation agent's model list)
        self.discussion_models = self.config.get(
            "discussion_models",
            ["gpt-4o", "claude-3-5-sonnet-latest", "deepseek-v3", "gemini-2.5-flash", "grok-4"]
        )
        
        # Read discussion parameters from config
        self.discussion_temperature = self.config.get("discussion_temperature", 0.1)
        # Voting (APPROVE/REJECT + 1-2 sentence reasoning) requires few tokens; modification tasks need full output space
        self.discussion_max_tokens   = self.config.get("discussion_max_tokens", 256)
        self.modification_max_tokens = self.config.get("modification_max_tokens", 2048)
        self.max_voting_rounds = self.config.get("max_voting_rounds", 3)
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Implement BaseAgent abstract method process
        
        Args:
            **kwargs: contains the following parameters:
                - generation_results: Dict - per-model generation results {model_name: {bhc: str, di: str, ...}}
                - verification_results: Dict - verification results {model_name: {overall_score: float, ...}}
                - verification_summary: Dict - verification summary info
                - shared_context: str - original shared context (for fact-checking by voting/revision models)
                - bhc_specific_context: str - BHC source data
                - di_specific_context: str - DI source data
        
        Returns:
            Final summary result
        """
        generation_results   = kwargs.get("generation_results", {})
        verification_results = kwargs.get("verification_results", {})
        verification_summary = kwargs.get("verification_summary", {})
        shared_context       = kwargs.get("shared_context", "")
        bhc_specific_context = kwargs.get("bhc_specific_context", "")
        di_specific_context  = kwargs.get("di_specific_context", "")
        
        # Step 1: Select the highest-scoring output
        best_model, best_result = self._select_best_output(
            generation_results, verification_results
        )
        
        if not best_model:
            return {
                "error": "No valid generation results found",
                "final_output": None
            }
        
        # Step 2: Collect secondary results and verification logs
        secondary_results = self._collect_secondary_results(
            generation_results, verification_results, best_model
        )
        
        # Step 3: Multi-model discussion
        consensus_result = self._multi_model_discussion(
            best_result=best_result,
            best_model=best_model,
            secondary_results=secondary_results,
            verification_summary=verification_summary,
            verification_results=verification_results,
            shared_context=shared_context,
            bhc_specific_context=bhc_specific_context,
            di_specific_context=di_specific_context,
        )
        
        return {
            "best_model": best_model,
            "best_score": verification_results.get(best_model, {}).get("overall_score", 0.0),
            "consensus_result": consensus_result,
            "discussion_models": self.discussion_models,
            "secondary_results": secondary_results
        }
    
    def _select_best_output(
        self,
        generation_results: Dict[str, Dict[str, Any]],
        verification_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Select highest-scoring output
        
        Returns:
            (best_model_name, best_result)
        """
        if not generation_results or not verification_results:
            return None, None
        
        # Find the model with the highest overall score (only from candidates in generation_results)
        best_model = None
        best_score = -1.0

        for model_name, verification in verification_results.items():
            if model_name not in generation_results:
                # This model was filtered upstream (e.g., API error), skip
                continue
            overall_score = verification.get("overall_score", 0.0)
            if overall_score > best_score:
                best_score = overall_score
                best_model = model_name

        # If no candidates in verification_results, fall back to the first entry in generation_results
        if not best_model:
            best_model = next(iter(generation_results), None)

        if not best_model:
            return None, None

        best_result = generation_results[best_model]
        return best_model, best_result
    
    def _collect_secondary_results(
        self,
        generation_results: Dict[str, Dict[str, Any]],
        verification_results: Dict[str, Dict[str, Any]],
        best_model: str
    ) -> List[Dict[str, Any]]:
        """
        Collect secondary results (sorted by score, excluding best model)
        
        Returns:
            List of runner-up results, each containing {model_name, bhc, di, score, verification}
        """
        # Sort all models by score (excluding the best model)
        model_scores = []
        for model_name, verification in verification_results.items():
            if model_name != best_model and model_name in generation_results:
                model_scores.append((
                    model_name,
                    verification.get("overall_score", 0.0),
                    generation_results[model_name],
                    verification
                ))
        
        # Sort by score in descending order
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Only take the top 3 secondary results (to avoid overly long discussion content)
        secondary_results = []
        for model_name, score, generation, verification in model_scores[:3]:
            secondary_results.append({
                "model_name": model_name,
                "bhc": generation.get("bhc", ""),
                "di": generation.get("di", ""),
                "score": score,
                "verification": verification
            })
        
        return secondary_results
    
    def _multi_model_discussion(
        self,
        best_result: Dict[str, Any],
        best_model: str,
        secondary_results: List[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        verification_results: Dict[str, Dict[str, Any]],
        shared_context: str = "",
        bhc_specific_context: str = "",
        di_specific_context: str = "",
    ) -> Dict[str, Any]:
        """
        Multi-model discussion: multi-round voting until consensus or max rounds
        
        Returns:
            {
                "final_bhc": str,
                "final_di": str,
                "votes": Dict[str, List[str]],  # {model_name: [round1_vote, round2_vote, ...]}
                "responses": Dict[str, List[str]],  # {model_name: [round1_response, round2_response, ...]}
                "approval_count": int,
                "rejection_count": int,
                "approved": bool,
                "voting_rounds": int,
                "all_rounds": List[Dict]  # voting results for all rounds
            }
        """
        current_result = best_result.copy()
        current_model = best_model
        all_votes = {model: [] for model in self.discussion_models}
        all_responses = {model: [] for model in self.discussion_models}
        all_rounds = []
        modification_log = []
        
        for round_num in range(1, self.max_voting_rounds + 1):
            self.log(f"Round {round_num} voting started...")
            
            # Build discussion prompt (for subsequent rounds, include the revised content)
            if round_num == 1:
                discussion_prompt = self._build_discussion_prompt(
                    current_result, current_model, secondary_results, verification_summary,
                    verification_results, shared_context, bhc_specific_context, di_specific_context,
                )
            else:
                # Subsequent rounds: include previous round voting results and revised content
                discussion_prompt = self._build_round_prompt(
                    current_result, current_model, all_rounds[-1], round_num
                )
            
            # System prompt (adjust strictness based on round number)
            if round_num == 1:
                system_prompt = """You are a medical expert conducting a rigorous peer review of a discharge summary.
Evaluate both the Brief Hospital Course (BHC) and Discharge Instructions (DI) impartially.

APPROVE if ALL of the following are met:
✓ Key clinical facts (diagnoses, procedures, medications) are accurate and consistent with the provided data
✓ No critical information is omitted (e.g., major diagnoses, key treatments, important safety warnings)
✓ BHC is concise and organized; DI is written in clear patient-facing language
✓ No contradictions or statements that could endanger patient safety

REJECT if ANY of the following apply:
✗ Factual errors: wrong drug, wrong dose, incorrect diagnosis, or fabricated clinical data not present in the source
✗ Critical omissions: missing a major diagnosis, key treatment, or essential discharge warning
✗ Logical contradictions that could confuse the patient or clinician
✗ BHC or DI is too vague or incomplete to be clinically useful

Be balanced: reject if there are meaningful clinical deficiencies, approve if the output is accurate and complete.

Respond ONLY with "APPROVE" or "REJECT" followed by a brief explanation (1-2 sentences).

Your response format:
APPROVE/REJECT
[Brief explanation]"""
            else:
                # Subsequent rounds: focus on whether key issues have been corrected
                system_prompt = f"""You are a medical expert reviewing a REVISED discharge summary (Round {round_num}).

Focus on whether the specific issues raised in the previous round have been corrected.

APPROVE if:
✓ Critical issues from Round {round_num - 1} have been addressed
✓ The output is now factually accurate and clinically complete
✓ Any remaining issues are minor and do not affect safety or usability

REJECT if:
✗ Previously identified factual errors or critical omissions remain uncorrected
✗ New errors were introduced in the revision
✗ The output is still clinically incomplete or misleading

Respond ONLY with "APPROVE" or "REJECT" followed by a brief explanation (1-2 sentences).

Your response format:
APPROVE/REJECT
[Brief explanation]"""
            
            # Collect votes from all models
            round_votes = {}
            round_responses = {}
            
            for model_name in self.discussion_models:
                try:
                    self.log(f"  Model {model_name} voting (round {round_num})...")
                    
                    response = self.llm_client.generate(
                        model_name=model_name,
                        prompt=discussion_prompt,
                        system_prompt=system_prompt,
                        temperature=self.discussion_temperature,
                        max_tokens=self.discussion_max_tokens
                    )
                    
                    round_responses[model_name] = response
                    all_responses[model_name].append(response)
                    
                    # Parse voting result
                    vote = self._parse_vote(response)
                    round_votes[model_name] = vote
                    all_votes[model_name].append(vote)
                    
                    self.log(f"  Model {model_name} vote: {vote}")
                    
                except Exception as e:
                    self.log(f"  Model {model_name} voting failed: {str(e)}", "ERROR")
                    round_votes[model_name] = "reject"  # Default rejection (conservative strategy)
                    round_responses[model_name] = f"Error: {str(e)}"
                    all_responses[model_name].append(f"Error: {str(e)}")
                    all_votes[model_name].append("reject")
            
            # Tally voting results
            vote_counts = Counter(round_votes.values())
            approval_count = vote_counts.get("approve", 0)
            rejection_count = vote_counts.get("reject", 0)
            
            # Record results for this round
            round_result = {
                "round": round_num,
                "votes": round_votes.copy(),
                "responses": round_responses.copy(),
                "approval_count": approval_count,
                "rejection_count": rejection_count
            }
            all_rounds.append(round_result)
            
            # Majority approval mechanism (subsequent rounds use more lenient criteria)
            if round_num == 1:
                # Round 1: strict criteria (majority approval)
                approved = approval_count > rejection_count
            else:
                # Subsequent rounds: if improvement is evident (approval count increased or equal), use more lenient criteria
                previous_approval = all_rounds[-2]["approval_count"] if len(all_rounds) > 1 else 0
                improvement = approval_count >= previous_approval
                
                if improvement:
                    # If improved, allow ties to pass (approval_count >= rejection_count)
                    approved = approval_count >= rejection_count
                    if approved and approval_count == rejection_count:
                        self.log(f"Round {round_num} vote: tie, but improved over previous round — passing", "INFO")
                else:
                    # If no improvement, still require majority approval
                    approved = approval_count > rejection_count
            
            self.log(f"Round {round_num} result: {approval_count} approve, {rejection_count} reject")
            
            # If approved, return result
            if approved:
                self.log(f"Round {round_num} vote passed!")
                return {
                    "final_bhc": current_result.get("bhc", ""),
                    "final_di": current_result.get("di", ""),
                    "votes": all_votes,
                    "responses": all_responses,
                    "approval_count": approval_count,
                    "rejection_count": rejection_count,
                    "approved": True,
                    "best_model": current_model,
                    "modified": round_num > 1,
                    "modification_log": modification_log,
                    "voting_rounds": round_num,
                    "all_rounds": all_rounds
                }
            
            # If not approved and max rounds not reached, proceed with modification
            if round_num < self.max_voting_rounds:
                self.log(f"Round {round_num} vote failed, starting revision...", "WARNING")
                modified_result = self._modify_output(
                    best_result=current_result,
                    best_model=current_model,
                    verification_results=verification_results,
                    secondary_results=secondary_results,
                    verification_summary=verification_summary,
                    votes=round_votes,
                    responses=round_responses,
                    round_num=round_num,
                    shared_context=shared_context,
                    bhc_specific_context=bhc_specific_context,
                    di_specific_context=di_specific_context,
                )
                current_result = {
                    "bhc": modified_result.get("final_bhc", ""),
                    "di": modified_result.get("final_di", "")
                }
                modification_log.extend(modified_result.get("modification_log", []))
                self.log(f"Revision complete, preparing for round {round_num + 1} voting...")
            else:
                # Max rounds reached without approval
                self.log(f"Reached max rounds ({self.max_voting_rounds}) without passing", "WARNING")
                break
        
        # All rounds failed, return the last modified result
        return {
            "final_bhc": current_result.get("bhc", ""),
            "final_di": current_result.get("di", ""),
            "votes": all_votes,
            "responses": all_responses,
            "approval_count": all_rounds[-1]["approval_count"] if all_rounds else 0,
            "rejection_count": all_rounds[-1]["rejection_count"] if all_rounds else 0,
            "approved": False,
            "best_model": current_model,
            "modified": True,
            "modification_log": modification_log,
            "voting_rounds": len(all_rounds),
            "all_rounds": all_rounds
        }
    
    def _build_discussion_prompt(
        self,
        best_result: Dict[str, Any],
        best_model: str,
        secondary_results: List[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        verification_results: Dict[str, Dict[str, Any]],
        shared_context: str = "",
        bhc_specific_context: str = "",
        di_specific_context: str = "",
    ) -> str:
        """
        Build voting prompt with original EHR data for fact-checking; hide internal verification scores (avoid anchoring).
        """
        best_bhc = best_result.get("bhc", "")
        best_di  = best_result.get("di", "")
        best_verification = verification_results.get(best_model, {})

        # ── Source input summary (truncated to control prompt length) ──────────────────────────
        src_block = ""
        if shared_context or bhc_specific_context or di_specific_context:
            sc_preview  = (shared_context[:400]   + "…") if len(shared_context)   > 400  else shared_context
            bhc_preview = (bhc_specific_context[:600] + "…") if len(bhc_specific_context) > 600 else bhc_specific_context
            di_preview  = (di_specific_context[:400]  + "…") if len(di_specific_context)  > 400 else di_specific_context
            src_block = f"""
**SOURCE CLINICAL DATA (use this to verify factual accuracy):**

[Shared — diagnoses, demographics]
{sc_preview}

[BHC Source — clinical notes, active issues, chronic issues]
{bhc_preview}

[DI Source — discharge medications, disposition, follow-up]
{di_preview}
"""

        prompt = f"""You are conducting a peer review of a hospital discharge summary.
{src_block}
**OUTPUT UNDER REVIEW:**

Brief Hospital Course (BHC):
{best_bhc}

Discharge Instructions (DI):
{best_di}
"""

        # Improvement suggestions (from automated verification)
        improvement_log = best_verification.get("improvement_log", [])
        if improvement_log:
            prompt += f"\n**AUTOMATED VERIFICATION FLAGS ({len(improvement_log)} items — consider but do not rely solely on these):**\n"
            for idx, item in enumerate(improvement_log[:5], 1):
                prompt += f"{idx}. [{item.get('type', 'unknown')}] {item.get('suggestion', '')}\n"
            if len(improvement_log) > 5:
                prompt += f"… and {len(improvement_log) - 5} more\n"

        # Secondary versions for comparison (scores hidden to avoid anchoring bias)
        if secondary_results:
            prompt += "\n**ALTERNATIVE VERSIONS (for reference only):**\n"
            for idx, alt in enumerate(secondary_results[:2], 1):
                prompt += f"\n--- Alternative {idx} ---\n"
                prompt += f"BHC:\n{alt['bhc'][:400]}…\n" if len(alt['bhc']) > 400 else f"BHC:\n{alt['bhc']}\n"
                prompt += f"DI:\n{alt['di'][:400]}…\n"  if len(alt['di'])  > 400 else f"DI:\n{alt['di']}\n"

        prompt += """
**YOUR TASK — evaluate against the source clinical data above:**
1. Factual accuracy: Do all diagnoses, drugs, doses, and procedures match the source data? Any invented facts?
2. Completeness: Are all major active problems covered in BHC? Are discharge medications listed correctly in DI?
3. BHC ↔ DI consistency: Does the DI align with the BHC? No contradictions?
4. Safety: Are warning signs specific and relevant? Are medication instructions correct?

**DECISION:**
APPROVE — if the output is factually accurate, complete, and safe.
REJECT  — if there are factual errors, critical omissions, or medication/safety issues.

Respond with ONLY "APPROVE" or "REJECT" on the first line, followed by one sentence of reasoning.

Example:
APPROVE
All key diagnoses and medications are correctly reflected from the source data with no omissions.

or

REJECT
The DI lists metformin 1000 mg but the source shows it was held during admission; also omits the INR monitoring instruction.
"""
        return prompt
    
    def _parse_vote(self, response: str) -> str:
        """
        Parse vote result conservatively: check REJECT on first line before APPROVE.
        Avoid misclassifying ambiguous replies like 'APPROVE with caveats but REJECT the DI' as approval.
        """
        first_line = response.strip().split("\n")[0].upper()
        # Check the first line first
        if "REJECT" in first_line:
            return "reject"
        if "APPROVE" in first_line:
            return "approve"
        # When first line is ambiguous, check the full response (conservative: reject if REJECT is found)
        full = response.upper()
        if "REJECT" in full:
            return "reject"
        if "APPROVE" in full:
            return "approve"
        self.log(f"Unable to parse vote result: {response[:100]}", "WARNING")
        return "reject"
    
    def _build_round_prompt(
        self,
        current_result: Dict[str, Any],
        current_model: str,
        previous_round: Dict[str, Any],
        round_num: int
    ) -> str:
        """
        Build discussion prompt for later rounds (previous votes + revised content)
        """
        current_bhc = current_result.get("bhc", "")
        current_di = current_result.get("di", "")
        
        prompt = f"""You are reviewing a REVISED discharge summary (Round {round_num}).

**PREVIOUS ROUND ({round_num - 1}) VOTING RESULTS:**
"""
        for model_name, vote in previous_round.get("votes", {}).items():
            response = previous_round.get("responses", {}).get(model_name, "")
            prompt += f"- {model_name}: {vote.upper()}\n"
            if response and len(response) < 200:
                prompt += f"  Reason: {response[:200]}\n"
        
        prompt += f"""
**CURRENT REVISED OUTPUT (Round {round_num}):**

Brief Hospital Course (BHC):
{current_bhc}

Discharge Instructions (DI):
{current_di}

**YOUR TASK:**
Review the revised output above. Consider:
1. Have the issues from the previous round been addressed?
2. Is the output now factually accurate and logically consistent?
3. Is the writing quality acceptable?

**DECISION:**
Respond with either "APPROVE" or "REJECT" followed by a brief explanation.

Example:
APPROVE
The revisions have addressed the previous concerns. The output is now acceptable.

or

REJECT
The output still contains critical issues: [specific problem]
"""
        return prompt
    
    def _modify_output(
        self,
        best_result: Dict[str, Any],
        best_model: str,
        verification_results: Dict[str, Dict[str, Any]],
        secondary_results: List[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        votes: Dict[str, str],
        responses: Dict[str, str],
        round_num: int = 1,
        shared_context: str = "",
        bhc_specific_context: str = "",
        di_specific_context: str = "",
    ) -> Dict[str, Any]:
        """
        Revise output: highest-scoring model applies verification log and suggestions
        
        Returns:
            {
                "final_bhc": str,
                "final_di": str,
                "modification_log": List[Dict]
            }
        """
        try:
            # Collect improvement suggestions and rejection reasons
            improvement_log = verification_results.get(best_model, {}).get("improvement_log", [])
            rejection_reasons = []
            
            # Extract rejection reasons from voting responses
            for model_name, response in responses.items():
                # Handle response format for multi-round voting
                if isinstance(response, list):
                    # Multi-round voting: use the last round's response
                    current_response = response[-1] if response else ""
                else:
                    current_response = response
                
                # Check if this model voted to reject
                model_vote = votes.get(model_name)
                if isinstance(model_vote, list):
                    # Multi-round voting: use the last round's vote
                    is_reject = model_vote[-1] == "reject" if model_vote else False
                else:
                    is_reject = model_vote == "reject"
                
                if is_reject:
                    # Extract rejection reason (skip the content after the "REJECT" keyword)
                    if isinstance(current_response, str):
                        lines = current_response.split('\n')
                        for line in lines[1:]:  # Skip the first line (usually "REJECT")
                            if line.strip():
                                rejection_reasons.append(f"{model_name}: {line.strip()}")
                                break
            
            # Build modification prompt
            modification_prompt = self._build_modification_prompt(
                best_result=best_result,
                best_model=best_model,
                improvement_log=improvement_log,
                rejection_reasons=rejection_reasons,
                secondary_results=secondary_results,
                verification_results=verification_results,
                shared_context=shared_context,
                bhc_specific_context=bhc_specific_context,
                di_specific_context=di_specific_context,
            )
            
            # System prompt (adjusted based on round number)
            if round_num == 1:
                system_prompt = """You are a medical expert tasked with improving a discharge summary based on verification feedback and peer review comments.

CRITICAL REQUIREMENTS:
- Address ALL critical issues mentioned in the improvement suggestions and rejection reasons
- Focus on fixing FACTUAL ERRORS and LOGICAL INCONSISTENCIES first
- Maintain factual accuracy - only modify based on evidence from the original EHR data
- Ensure logical consistency between Brief Hospital Course (BHC) and Discharge Instructions (DI)
- Preserve the professional medical writing style
- Make targeted improvements - fix what needs to be fixed, don't rewrite everything
- If an issue cannot be fixed with available information, note it but do not invent information

Your task is to revise the discharge summary to address the identified issues while maintaining its core content and accuracy."""
            else:
                # Subsequent rounds: more focused on addressing specific issues from the previous round
                system_prompt = f"""You are a medical expert revising a discharge summary based on Round {round_num - 1} feedback.

CRITICAL REQUIREMENTS:
- Focus on addressing the SPECIFIC issues mentioned in the previous round's rejection reasons
- Make targeted fixes - don't make unnecessary changes
- Ensure the fixes actually address the concerns raised
- Maintain factual accuracy - only use information from the original EHR data
- Keep the output concise and focused
- If previous concerns were about specific sentences or sections, fix those specific parts

Your task is to make targeted improvements that directly address the feedback from Round {round_num - 1}."""
            
            # Use the highest-scoring model for modification
            self.log(f"Using model {best_model} for revision...")
            
            # Modify BHC and DI separately
            modified_bhc = self._modify_bhc(
                original_bhc=best_result.get("bhc", ""),
                modification_prompt=modification_prompt,
                improvement_log=improvement_log,
                rejection_reasons=rejection_reasons,
                model_name=best_model,
                system_prompt=system_prompt
            )
            
            modified_di = self._modify_di(
                original_di=best_result.get("di", ""),
                modification_prompt=modification_prompt,
                improvement_log=improvement_log,
                rejection_reasons=rejection_reasons,
                model_name=best_model,
                system_prompt=system_prompt
            )
            
            modification_log = []
            if improvement_log:
                modification_log.append({
                    "type": "improvement_suggestions",
                    "count": len(improvement_log),
                    "addressed": True
                })
            if rejection_reasons:
                modification_log.append({
                    "type": "rejection_reasons",
                    "reasons": rejection_reasons,
                    "addressed": True
                })
            
            return {
                "final_bhc": modified_bhc,
                "final_di": modified_di,
                "modification_log": modification_log
            }
            
        except Exception as e:
            self.log(f"Output revision failed: {str(e)}", "ERROR")
            # If modification fails, return the original output
            return {
                "final_bhc": best_result.get("bhc", ""),
                "final_di": best_result.get("di", ""),
                "modification_log": [{"type": "error", "message": str(e)}]
            }
    
    def _build_modification_prompt(
        self,
        best_result: Dict[str, Any],
        best_model: str,
        improvement_log: List[Dict[str, Any]],
        rejection_reasons: List[str],
        secondary_results: List[Dict[str, Any]],
        verification_results: Dict[str, Dict[str, Any]],
        shared_context: str = "",
        bhc_specific_context: str = "",
        di_specific_context: str = "",
    ) -> str:
        """Build modification prompt including original EHR data as the basis for edits."""
        best_bhc = best_result.get("bhc", "")
        best_di  = best_result.get("di", "")

        # ── Source input (truncated) ───────────────────────────────────────────────
        src_block = ""
        if shared_context or bhc_specific_context or di_specific_context:
            sc_preview  = (shared_context[:400]       + "…") if len(shared_context)       > 400  else shared_context
            bhc_preview = (bhc_specific_context[:800] + "…") if len(bhc_specific_context) > 800  else bhc_specific_context
            di_preview  = (di_specific_context[:600]  + "…") if len(di_specific_context)  > 600  else di_specific_context
            src_block = f"""
**SOURCE CLINICAL DATA (ground truth — base ALL corrections on this):**

[Shared — diagnoses, demographics]
{sc_preview}

[BHC Source — clinical notes, active issues]
{bhc_preview}

[DI Source — discharge medications, disposition, follow-up]
{di_preview}
"""

        prompt = f"""You need to revise a discharge summary that was rejected during peer review.
{src_block}
**CURRENT OUTPUT TO REVISE:**

Brief Hospital Course (BHC):
{best_bhc}

Discharge Instructions (DI):
{best_di}
"""

        if improvement_log:
            prompt += f"\n**AUTOMATED VERIFICATION FLAGS ({len(improvement_log)} items):**\n"
            for idx, item in enumerate(improvement_log, 1):
                prompt += f"{idx}. [{item.get('type', 'unknown')}] {item.get('suggestion', '')}\n"
                if "sentence" in item:
                    prompt += f"   Flagged text: {item['sentence'][:200]}\n"

        if rejection_reasons:
            prompt += "\n**PEER REVIEW REJECTION REASONS (highest priority to fix):**\n"
            for idx, reason in enumerate(rejection_reasons, 1):
                prompt += f"{idx}. {reason}\n"

        if secondary_results:
            prompt += "\n**ALTERNATIVE VERSIONS (for reference — do not copy wholesale):**\n"
            for idx, alt in enumerate(secondary_results[:2], 1):
                prompt += f"\n--- Alternative {idx} ---\n"
                prompt += f"BHC: {alt['bhc'][:300]}…\n" if len(alt['bhc']) > 300 else f"BHC: {alt['bhc']}\n"
                prompt += f"DI: {alt['di'][:300]}…\n"  if len(alt['di'])  > 300 else f"DI: {alt['di']}\n"

        return prompt
    
    def _modify_bhc(
        self,
        original_bhc: str,
        modification_prompt: str,
        improvement_log: List[Dict[str, Any]],
        rejection_reasons: List[str],
        model_name: str,
        system_prompt: str,
        round_num: int = 1
    ) -> str:
        """
        Revise BHC
        """
        # Filter improvement suggestions relevant to BHC
        bhc_improvements = [
            item for item in improvement_log
            if item.get('type') == 'factual_hallucination' or 'bhc' in item.get('suggestion', '').lower()
        ]
        
        if not bhc_improvements and not rejection_reasons:
            # If no BHC-related improvement suggestions, return original content
            return original_bhc
        
        prompt = f"""{modification_prompt}

**YOUR TASK - REVISE BHC:**
Revise the Brief Hospital Course (BHC) above to address the following issues:

"""
        
        if bhc_improvements:
            prompt += "**BHC-Specific Issues:**\n"
            for item in bhc_improvements:
                prompt += f"- {item.get('suggestion', '')}\n"
        
        if rejection_reasons:
            prompt += "\n**General Issues from Peer Review:**\n"
            for reason in rejection_reasons:
                prompt += f"- {reason}\n"
        
        prompt += f"""

**REQUIREMENTS:**
- Fix the SPECIFIC issues mentioned above
- Focus on CRITICAL problems first (factual errors, safety issues)
- Ensure all medical information is accurate and supported by evidence
- Maintain professional medical writing style
- Keep the chronological structure
- Do NOT invent information that is not in the original context
- Make targeted fixes - don't rewrite the entire BHC unless necessary
- Do NOT increase the output length by more than 20% compared to the original BHC
- If a rejection reason is vague or cannot be resolved with available information, preserve the original text for that section
- Output ONLY the revised BHC text. Do NOT include any headers like "Revised BHC:", "Key Changes Made:", or any explanatory notes about what was changed.

**OUTPUT (revised BHC text only, no headers or commentary):**"""
        
        try:
            modified_bhc = self.llm_client.generate(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.discussion_temperature,
                max_tokens=self.modification_max_tokens,
            )
            return _strip_modification_metadata(modified_bhc, "bhc") if modified_bhc else original_bhc
        except Exception as e:
            self.log(f"BHC revision failed: {str(e)}, using original content", "ERROR")
            return original_bhc
    
    def _modify_di(
        self,
        original_di: str,
        modification_prompt: str,
        improvement_log: List[Dict[str, Any]],
        rejection_reasons: List[str],
        model_name: str,
        system_prompt: str,
        round_num: int = 1
    ) -> str:
        """
        Revise DI
        """
        # Filter improvement suggestions relevant to DI
        di_improvements = [
            item for item in improvement_log
            if item.get('type') == 'logical_hallucination' or 'di' in item.get('suggestion', '').lower()
        ]
        
        if not di_improvements and not rejection_reasons:
            # If no DI-related improvement suggestions, return original content
            return original_di
        
        prompt = f"""{modification_prompt}

**YOUR TASK - REVISE DI:**
Revise the Discharge Instructions (DI) above to address the following issues:

"""
        
        if di_improvements:
            prompt += "**DI-Specific Issues:**\n"
            for item in di_improvements:
                prompt += f"- {item.get('suggestion', '')}\n"
        
        if rejection_reasons:
            prompt += "\n**General Issues from Peer Review:**\n"
            for reason in rejection_reasons:
                prompt += f"- {reason}\n"
        
        prompt += f"""

**REQUIREMENTS:**
- Fix the SPECIFIC issues mentioned above
- Focus on CRITICAL problems first (logical inconsistencies, medication errors, safety issues)
- Ensure medication instructions are accurate and clear
- Maintain patient-friendly language
- Keep instructions actionable and specific
- Do NOT invent information that is not in the original context
- Make targeted fixes - don't rewrite the entire DI unless necessary
- Do NOT increase the output length by more than 20% compared to the original DI
- If a rejection reason is vague or cannot be resolved with available information, preserve the original text for that section
- Output ONLY the revised DI text. Do NOT include any headers like "Revised DI:", "Key Revisions:", or any explanatory notes about what was changed.

**OUTPUT (revised DI text only, no headers or commentary):**"""
        
        try:
            modified_di = self.llm_client.generate(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.discussion_temperature,
                max_tokens=self.modification_max_tokens,
            )
            return _strip_modification_metadata(modified_di, "di") if modified_di else original_di
        except Exception as e:
            self.log(f"DI revision failed: {str(e)}, using original content", "ERROR")
            return original_di

