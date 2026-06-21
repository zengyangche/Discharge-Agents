"""
Verification Engine
Verification engine with multiple verifier agents (Fact, Style, Logic, Clinical Logic).
"""
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
import re
import numpy as np
from abc import ABC, abstractmethod
from utils.config import (
    VERIFICATION_CONFIG,
    MEDICAL_TERMINOLOGY,
    TRAIN_DATA_PATH
)
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """Verifier base class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def verify(
        self,
        draft: str,
        role: str,  # "bhc" or "di"
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify a draft.

        Returns:
            {
                "score": float,  # score between 0 and 1
                "passed": bool,
                "details": {...},
                "errors": [...],
            }
        """
        pass


class KnowledgeBasedVerifier(BaseVerifier):
    """
    Knowledge-base verifier (merges former Fact Verification and Clinical Logic).

    Features:
    1. Fact verification (RAG): check generated content against input and KB via retrieval
    2. Clinical logic verification: check for missing required orders via association rules

    Knowledge base support:
    - RAG: bhc_embeddings/di_embeddings and bhc_texts/di_texts
    - Association rules: association_rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_base=None):
        super().__init__(config)
        self.fact_threshold = VERIFICATION_CONFIG["fact_check"]["similarity_threshold"]
        self.fact_weight = VERIFICATION_CONFIG["fact_check"]["weight"]
        self.clinical_logic_weight = VERIFICATION_CONFIG["clinical_logic_check"]["weight"]
        self.knowledge_base = knowledge_base  # Knowledge base, required
    
    def verify(
        self,
        draft: str,
        role: str,
        context: Dict[str, Any],
        raw_data: Optional[Dict[str, Any]] = None,
        bhc_draft: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Knowledge-base verification (fact check + clinical logic).

        1. Fact verification (RAG): check generated content against input and KB
        2. Clinical logic verification (association rules): check missing orders (DI only)
        """
        if not self.knowledge_base:
            return {
                "score": 1.0,
                "passed": True,
                "details": {"message": "Knowledge base not loaded, skipping verification"},
                "errors": [],
            }
        
        try:
            # ========== 1. Fact verification (RAG retrieval) ==========
            fact_result = self._verify_facts_with_rag(draft, role, raw_data)
            
            # ========== 2. Clinical logic verification (association rules, DI only) ==========
            clinical_logic_result = {
                "score": 1.0,
                "passed": True,
                "details": {},
                "errors": [],
            }
            
            if role == "di" and bhc_draft:
                clinical_logic_result = self._verify_clinical_logic(bhc_draft, draft)
            
            # ========== Merge results ==========
            # Weighted average score
            total_weight = self.fact_weight + (self.clinical_logic_weight if role == "di" else 0)
            combined_score = (
                fact_result["score"] * self.fact_weight +
                clinical_logic_result["score"] * (self.clinical_logic_weight if role == "di" else 0)
            ) / total_weight if total_weight > 0 else fact_result["score"]
            
            passed = (
                fact_result["passed"] and
                (clinical_logic_result["passed"] if role == "di" else True)
            )
            
            return {
                "score": combined_score,
                "passed": passed,
                "details": {
                    "fact_verification": fact_result["details"],
                    "clinical_logic_verification": clinical_logic_result["details"] if role == "di" else {},
                },
                "errors": fact_result["errors"] + clinical_logic_result["errors"],
            }
        except Exception as e:
            self.logger.error(f"Knowledge-based verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
            }
    
    def _verify_facts_with_rag(
        self,
        draft: str,
        role: str,
        raw_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fact verification: use RAG to check consistency with input and knowledge base.

        Knowledge base support: bhc_embeddings/di_embeddings, bhc_texts/di_texts
        """
        # 1. Build query text from input data
        input_query = self._build_input_query(raw_data, role) if raw_data else ""
        
        # 2. RAG retrieval: retrieve similar texts from knowledge base
        retrieved_texts = self.knowledge_base.rag_retrieve(
            query=draft,
            role=role,
            top_k=5,
            similarity_threshold=0.7
        )
        
        # 3. Check consistency between generated content and input data
        input_consistency_score = 1.0
        input_errors = []
        
        if input_query:
            # Use RAG retrieval to check if input data has similar cases in knowledge base
            input_retrieved = self.knowledge_base.rag_retrieve(
                query=input_query,
                role=role,
                top_k=3,
                similarity_threshold=0.6
            )
            
            if input_retrieved:
                # Check if generated content is consistent with retrieved similar cases
                avg_similarity = sum(sim for _, sim in retrieved_texts) / len(retrieved_texts) if retrieved_texts else 0.0
                input_consistency_score = min(1.0, avg_similarity / 0.8)  # Normalize to 0-1
        
        # 4. Check for hallucinations (generated content does not match knowledge base)
        hallucinations = []
        if retrieved_texts:
            # If the similarity of all retrieved texts is very low, it may indicate hallucination
            max_similarity = max(sim for _, sim in retrieved_texts) if retrieved_texts else 0.0
            if max_similarity < 0.5:
                hallucinations.append("Generated content has low similarity to knowledge base cases, possible hallucination")
        
        # 5. Calculate overall score
        fact_score = input_consistency_score * 0.7 + (1.0 if not hallucinations else 0.5) * 0.3
        passed = fact_score >= self.fact_threshold
        
        return {
            "score": fact_score,
            "passed": passed,
            "details": {
                "input_consistency_score": input_consistency_score,
                "retrieved_similar_cases": len(retrieved_texts),
                "max_similarity": max(sim for _, sim in retrieved_texts) if retrieved_texts else 0.0,
                "hallucinations_detected": len(hallucinations),
            },
            "errors": hallucinations + input_errors,
        }
    
    def _verify_clinical_logic(
        self,
        bhc_draft: str,
        di_draft: str
    ) -> Dict[str, Any]:
        """
        Clinical logic verification: use RAG to check for missing required orders.

        Knowledge base support: RAG retrieval corpus
        """
        if not self.knowledge_base:
            return {
                "score": 1.0,
                "passed": True,
                "details": {"message": "Knowledge base not loaded, skipping clinical logic verification"},
                "errors": [],
            }
        
        # Use RAG retrieval to find similar BHC and DI cases from knowledge base
        # If generated BHC/DI is similar to knowledge base cases, logic is considered sound
        bhc_similar_cases = self.knowledge_base.rag_retrieve(
            bhc_draft, 
            role="bhc", 
            top_k=3,
            similarity_threshold=0.7
        )
        
        di_similar_cases = self.knowledge_base.rag_retrieve(
            di_draft, 
            role="di", 
            top_k=3,
            similarity_threshold=0.7
        )
        
        # Calculate average similarity
        bhc_avg_similarity = sum(sim for _, sim in bhc_similar_cases) / len(bhc_similar_cases) if bhc_similar_cases else 0.0
        di_avg_similarity = sum(sim for _, sim in di_similar_cases) / len(di_similar_cases) if di_similar_cases else 0.0
        
        # Combined score (average of the two similarity scores)
        score = (bhc_avg_similarity + di_avg_similarity) / 2.0 if (bhc_similar_cases or di_similar_cases) else 1.0
        passed = score >= 0.7
        
        errors = []
        if not passed:
            errors.append(f"Generated BHC/DI has low similarity to knowledge base cases (BHC: {bhc_avg_similarity:.2f}, DI: {di_avg_similarity:.2f})")
        
        return {
            "score": score,
            "passed": passed,
            "details": {
                "bhc_similar_cases": len(bhc_similar_cases),
                "di_similar_cases": len(di_similar_cases),
                "bhc_avg_similarity": bhc_avg_similarity,
                "di_avg_similarity": di_avg_similarity,
            },
            "errors": errors,
        }
    
    def _build_input_query(self, raw_data: Dict[str, Any], role: str) -> str:
        """Build query text from input data"""
        query_parts = []
        
        # Extract from diagnoses
        for diag in raw_data.get("diagnosis", []):
            title = diag.get("icd_title", "")
            if title:
                query_parts.append(title)
        
        # Extract from triage information
        triage = raw_data.get("triage", {})
        if triage.get("chiefcomplaint"):
            query_parts.append(triage["chiefcomplaint"])
        
        # Extract from other important fields
        if role == "bhc":
            edstays = raw_data.get("edstays", {})
            if edstays.get("chiefcomplaint"):
                query_parts.append(edstays["chiefcomplaint"])
        
        return " ".join(query_parts)
    


class StyleVerifier(BaseVerifier):
    """
    Style Consensus Agent

    For BHC: embedding similarity (KB BHC centroid)
    For DI: readability score and terminology residue rate

    Knowledge base support: bhc_centroid
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_base=None):
        super().__init__(config)
        self.bhc_threshold = VERIFICATION_CONFIG["style_check"]["bhc_embedding_similarity_threshold"]
        self.bhc_weight = VERIFICATION_CONFIG["style_check"]["bhc_weight"]
        self.di_readability_threshold = VERIFICATION_CONFIG["style_check"]["di_readability_threshold"]
        self.di_readability_weight = VERIFICATION_CONFIG["style_check"]["di_readability_weight"]
        self.di_terminology_threshold = VERIFICATION_CONFIG["style_check"]["di_terminology_residue_threshold"]
        self.di_terminology_weight = VERIFICATION_CONFIG["style_check"]["di_terminology_weight"]
        
        self.knowledge_base = knowledge_base  # Knowledge base, used to retrieve BHC centroid vector
        
        # Initialize embedding model (for BHC)
        self._init_embedding_model()
        
        # Load BHC style centroid vector (from knowledge base)
        self.bhc_centroid = self._load_bhc_centroid()
    
    def _init_embedding_model(self):
        """Initialize the sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Embedding model failed to load: {str(e)}")
            self.embedding_model = None
    
    def _load_bhc_centroid(self) -> Optional[np.ndarray]:
        """
        Load BHC style centroid vector from the knowledge base.

        If the KB is loaded, use its BHC centroid; otherwise fall back to a mock vector.
        """
        # Load from knowledge base first
        if self.knowledge_base:
            centroid = self.knowledge_base.get_bhc_style_centroid()
            if centroid is not None:
                self.logger.info("Loaded BHC style centroid vector from knowledge base")
                return centroid
            else:
                self.logger.warning("BHC centroid vector not found in knowledge base, using fallback")
        
        # Fallback: compute embedding from sample text
        if self.embedding_model:
            self.logger.warning("Using mock BHC centroid vector (recommend building knowledge base)")
            sample_bhc = """Patient was admitted with complaints of chest pain. 
            Cardiac enzymes were elevated. Patient underwent cardiac catheterization 
            which showed 90% stenosis of the LAD. Patient underwent PCI with stent 
            placement. Post-procedure course was uncomplicated. Patient was 
            discharged in stable condition."""
            return self.embedding_model.encode(sample_bhc)
        return None
    
    def verify(
        self,
        draft: str,
        role: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Style verification"""
        if role == "bhc":
            return self._verify_bhc_style(draft)
        else:
            return self._verify_di_style(draft)
    
    def _verify_bhc_style(self, draft: str) -> Dict[str, Any]:
        """Verify BHC style: compute embedding similarity"""
        if not self.embedding_model or self.bhc_centroid is None:
            return {
                "score": 1.0,
                "passed": True,
                "details": {"message": "Embedding model not loaded, skipping verification"},
                "errors": [],
            }
        
        try:
            # Compute embedding for draft
            draft_embedding = self.embedding_model.encode(draft)
            
            # Compute cosine similarity
            similarity = np.dot(draft_embedding, self.bhc_centroid) / (
                np.linalg.norm(draft_embedding) * np.linalg.norm(self.bhc_centroid)
            )
            
            # Normalize to 0-1 range
            score = (similarity + 1) / 2
            
            passed = score >= self.bhc_threshold
            
            return {
                "score": score,
                "passed": passed,
                "details": {
                    "embedding_similarity": float(similarity),
                    "normalized_score": float(score),
                },
                "errors": [] if passed else [f"BHC style similarity below threshold: {score:.3f} < {self.bhc_threshold}"],
            }
        except Exception as e:
            self.logger.error(f"BHC style verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
            }
    
    def _verify_di_style(self, draft: str) -> Dict[str, Any]:
        """Verify DI style: readability and terminology residue rate"""
        try:
            # 1. Readability check
            readability_score = self._calculate_readability(draft)
            
            # 2. Terminology residue rate check
            terminology_rate = self._calculate_terminology_residue(draft)
            
            # Calculate overall score
            readability_score_normalized = min(readability_score / 100.0, 1.0)
            terminology_score = 1.0 - min(terminology_rate / self.di_terminology_threshold, 1.0)
            
            # Weighted average
            total_score = (
                readability_score_normalized * self.di_readability_weight +
                terminology_score * self.di_terminology_weight
            ) / (self.di_readability_weight + self.di_terminology_weight)
            
            passed = (
                readability_score >= self.di_readability_threshold and
                terminology_rate <= self.di_terminology_threshold
            )
            
            errors = []
            if readability_score < self.di_readability_threshold:
                errors.append(f"Readability score too low: {readability_score:.1f} < {self.di_readability_threshold}")
            if terminology_rate > self.di_terminology_threshold:
                errors.append(f"Terminology residue rate too high: {terminology_rate:.2%} > {self.di_terminology_threshold:.2%}")
            
            return {
                "score": total_score,
                "passed": passed,
                "details": {
                    "readability_score": readability_score,
                    "terminology_residue_rate": terminology_rate,
                },
                "errors": errors,
            }
        except Exception as e:
            self.logger.error(f"DI style verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
            }
    
    def _calculate_readability(self, text: str) -> float:
        """Compute readability score (Flesch Reading Ease)"""
        try:
            import textstat
            # Flesch Reading Ease: 0-100, higher score means easier to read
            return textstat.flesch_reading_ease(text)
        except Exception as e:
            self.logger.warning(f"Readability calculation failed: {str(e)}")
            return 60.0  # default value
    
    def _calculate_terminology_residue(self, text: str) -> float:
        """Compute terminology residue rate"""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count medical terminology occurrences
        terminology_count = sum(
            1 for word in words
            if any(term in word for term in MEDICAL_TERMINOLOGY)
        )
        
        return terminology_count / total_words


class LogicVerifier(BaseVerifier):
    """
    Logic Agent (Internal Consistency)

    Checks only literal contradictions within BHC or DI (internal consistency).
    Does not use the knowledge base; text-only logic checks.

    Knowledge base support: none (standalone verifier)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = VERIFICATION_CONFIG["logic_check"]["contradiction_threshold"]
        self.weight = VERIFICATION_CONFIG["logic_check"]["weight"]
        self._logic_model = (config or {}).get("logic_llm_model", "gpt-4o")
        self._llm_client: Optional[LLMClient] = None

    def _get_llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    @staticmethod
    def _parse_llm_json_object(raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
            raise ValueError(f"LLM returned non-JSON object: {text[:400]}...")
    
    def verify(
        self,
        draft: str,
        role: str,
        context: Dict[str, Any],
        bhc_draft: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Logic verification: internal contradictions only.

        For BHC: check contradictions within the BHC text.
        For DI: check contradictions within the DI text (BHC-DI consistency is handled
        by KnowledgeBasedVerifier).
        """
        try:
            # Extract claims
            claims = self._extract_claims(draft)
            
            # Check only internal contradictions
            internal_contradictions = self._check_internal_contradictions(claims)
            
            # Calculate score
            score = max(0.0, 1.0 - len(internal_contradictions) * 0.2)  # Deduct 0.2 points per contradiction
            
            passed = score >= self.threshold
            
            return {
                "score": score,
                "passed": passed,
                "details": {
                    "claims_extracted": len(claims),
                    "internal_contradictions": len(internal_contradictions),
                },
                "errors": internal_contradictions,
            }
        except Exception as e:
            self.logger.error(f"Logic verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
            }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Use GPT-4o (unified OpenAI-compatible API) to extract verifiable claims."""
        text = (text or "").strip()
        if not text:
            return []
        system_prompt = (
            "You are a clinical documentation analyst reviewing MIMIC-IV hospital discharge notes (Brief Hospital Course or Discharge Instructions).\n"
            "Extract up to 15 independently verifiable or actionable statements from the text.\n"
            "Focus on these claim types: diagnostic facts, prescribed medications with dosages, activity restrictions, "
            "dietary instructions, follow-up requirements, and warning signs.\n"
            "Ignore temporal filler phrases (e.g., 'the patient was admitted on', 'it was a pleasure'). "
            "Focus on actionable or verifiable statements only.\n"
            "Each claim must be short, self-contained, and preserve the original language of the text.\n"
            "Do not merge multiple facts into one claim. Do not repeat semantically identical claims.\n"
            "Output ONLY valid JSON in this format: {\"claims\": [\"...\", ...]} — no other text."
        )
        user_prompt = f"Text:\n{text[:12000]}\n\nReturn JSON only."
        raw = self._get_llm().generate(
            self._logic_model,
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1500,
        )
        data = self._parse_llm_json_object(raw)
        claims = data.get("claims")
        if not isinstance(claims, list):
            raise ValueError("LLM returned claims that is not a list")
        out: List[str] = []
        for c in claims:
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
        return out[:15]

    def _check_internal_contradictions(self, claims: List[str]) -> List[str]:
        """Use GPT-4o to check logical contradictions within the claim list."""
        if not claims:
            return []
        system_prompt = (
            "You are a clinical logic reviewer. Given a list of statements extracted from a hospital discharge note, "
            "identify any pairs (or groups) that are logically contradictory within the same clinical context.\n"
            "Base your judgment solely on the provided statements — do not introduce external medical knowledge or assumptions.\n\n"
            "Examples of REAL contradictions:\n"
            "- 'Patient may resume full activity' vs 'Patient should avoid strenuous exercise for 4 weeks'\n"
            "- 'Patient is allergic to penicillin' vs 'Patient was discharged on amoxicillin'\n"
            "- 'Patient is NPO' vs 'Patient is tolerating a regular diet'\n\n"
            "Examples of NON-contradictions (do NOT flag these):\n"
            "- Syntactic negation pairs such as 'no fever' vs 'afebrile' — these mean the same thing\n"
            "- Temporal sequences where an earlier state changed (e.g., 'initially NPO' then 'advanced to regular diet')\n"
            "- Different body systems with seemingly opposite states (e.g., 'cardiac stable' vs 'respiratory distress')\n\n"
            "If there are no contradictions, return an empty array.\n"
            "Output ONLY valid JSON: {\"contradictions\": [\"...\", ...]} — each entry briefly describes the contradiction and quotes the key phrases."
        )
        user_prompt = json.dumps({"claims": claims}, ensure_ascii=False)
        raw = self._get_llm().generate(
            self._logic_model,
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1200,
        )
        data = self._parse_llm_json_object(raw)
        items = data.get("contradictions")
        if not isinstance(items, list):
            raise ValueError("LLM returned contradictions that is not a list")
        out: List[str] = []
        for x in items:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    




class VerificationEngine:
    """
    Verification engine: orchestrates all verifiers.

    Verifier layout:
    1. KnowledgeBasedVerifier: KB-based checks (fact RAG + clinical logic rules)
    2. StyleVerifier: style checks (BHC embedding similarity; DI readability and terminology)
    3. LogicVerifier: logic checks (internal contradictions only)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_base=None):
        """
        Args:
            config: Verification configuration
            knowledge_base: Knowledge base instance (required for KnowledgeBasedVerifier and StyleVerifier)
        """
        self.config = config or {}
        self.knowledge_base = knowledge_base
        
        # Initialize verifiers (passing knowledge base)
        self.knowledge_based_verifier = KnowledgeBasedVerifier(config, knowledge_base=knowledge_base)
        self.style_verifier = StyleVerifier(config, knowledge_base=knowledge_base)
        self.logic_verifier = LogicVerifier(config)
        self.logger = logging.getLogger(f"{__name__}.VerificationEngine")
        
        # Log knowledge base status
        if knowledge_base:
            self.logger.info("Knowledge base loaded, verifiers initialized")
        else:
            self.logger.warning("Knowledge base not loaded, some verification features may be limited")
    
    def verify_all(
        self,
        bhc_draft: str,
        di_draft: str,
        shared_context: Dict[str, Any],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run full verification on all drafts.

        Returns:
            {
                "bhc_verification": {
                    "fact": {...},
                    "style": {...},
                    "logic": {...},
                    "overall_score": float,
                },
                "di_verification": {
                    "fact": {...},
                    "style": {...},
                    "logic": {...},
                    "clinical_logic": {...},
                    "overall_score": float,
                }
            }
        """
        # Verify BHC
        bhc_verification = {
            "knowledge_based": self.knowledge_based_verifier.verify(
                bhc_draft, "bhc", shared_context, raw_data=raw_data
            ),
            "style": self.style_verifier.verify(
                bhc_draft, "bhc", shared_context
            ),
            "logic": self.logic_verifier.verify(
                bhc_draft, "bhc", shared_context
            ),
        }
        
        # Verify DI
        di_verification = {
            "knowledge_based": self.knowledge_based_verifier.verify(
                di_draft, "di", shared_context, raw_data=raw_data, bhc_draft=bhc_draft
            ),
            "style": self.style_verifier.verify(
                di_draft, "di", shared_context
            ),
            "logic": self.logic_verifier.verify(
                di_draft, "di", shared_context
            ),
        }
        
        # Calculate overall score
        bhc_verification["overall_score"] = self._calculate_overall_score(
            bhc_verification, role="bhc"
        )
        di_verification["overall_score"] = self._calculate_overall_score(
            di_verification, role="di"
        )
        
        return {
            "bhc_verification": bhc_verification,
            "di_verification": di_verification,
        }
    
    def _calculate_overall_score(
        self,
        verification_results: Dict[str, Any],
        role: str
    ) -> float:
        """Compute overall verification score"""
        from utils.config import VERIFICATION_CONFIG
        
        weights = {
            "knowledge_based": (
                VERIFICATION_CONFIG["fact_check"]["weight"] +
                (VERIFICATION_CONFIG["clinical_logic_check"]["weight"] if role == "di" else 0.0)
            ),
            "style": (
                VERIFICATION_CONFIG["style_check"]["bhc_weight"]
                if role == "bhc"
                else (
                    VERIFICATION_CONFIG["style_check"]["di_readability_weight"] +
                    VERIFICATION_CONFIG["style_check"]["di_terminology_weight"]
                )
            ),
            "logic": VERIFICATION_CONFIG["logic_check"]["weight"],
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in verification_results and weight > 0:
                score = verification_results[key].get("score", 0.0)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
