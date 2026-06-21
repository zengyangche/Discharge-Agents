"""
Verification Agent: implements three verification mechanisms
1. Factual hallucination check: verify BHC against source input for factual errors
2. Logical hallucination check: use QA pairs to verify logical consistency between BHC and DI
3. Writing style verification: compute similarity to average BHC/DI embeddings from training set
"""
from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent
import logging
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class VerificationAgent(BaseAgent):
    """
    Verification Agent: runs three verifications on generated BHC and DI
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_base=None):
        super().__init__("VerificationAgent", config)
        self.knowledge_base = knowledge_base
        
        # Read weights and thresholds from config
        verification_config = self.config.get("verification", {})
        self.weights = verification_config.get("weights", {
            "fact": 0.5,
            "logic": 0.3,
            "style": 0.2
        })
        self.thresholds = verification_config.get("thresholds", {
            "fact": 0.5,
            "logic": 0.5,
            "style": 0.5
        })
        
        self.logger.info(f"Verification config - weights: {self.weights}, thresholds: {self.thresholds}")
        
        # Initialize embedding model (used for style verification)
        self._init_embedding_model()
        
        # Load average embeddings for BHC and DI (used for style verification)
        self.bhc_centroid = None
        self.di_centroid = None
        self._load_style_centroids()
        
        # Verification log storage
        self.verification_logs = []
    
    def _init_embedding_model(self):
        """Initialize the sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Embedding model failed to load: {str(e)}")
            self.embedding_model = None
    
    def _load_style_centroids(self):
        """Load style-verification vectors: prefer top-K training embeddings, fall back to centroid."""
        self.bhc_train_embeddings = None
        self.di_train_embeddings  = None

        if self.knowledge_base:
            self.bhc_centroid = self.knowledge_base.bhc_centroid
            self.di_centroid  = self.knowledge_base.di_centroid
            # Try to load complete training set embeddings (for top-K style scoring)
            self.bhc_train_embeddings = getattr(self.knowledge_base, "bhc_embeddings", None)
            self.di_train_embeddings  = getattr(self.knowledge_base, "di_embeddings",  None)
            if self.bhc_train_embeddings is not None:
                self.logger.info(f"Loaded BHC training embeddings ({len(self.bhc_train_embeddings)} samples), top-K style scoring enabled")
            elif self.bhc_centroid is not None:
                self.logger.info("Loaded BHC style centroid vector (fallback)")
            if self.di_train_embeddings is not None:
                self.logger.info(f"Loaded DI training embeddings ({len(self.di_train_embeddings)} samples), top-K style scoring enabled")
            elif self.di_centroid is not None:
                self.logger.info("Loaded DI style centroid vector (fallback)")
        else:
            kb_dir = Path("outputs/knowledge_base")
            for role, attr_c, attr_e, fname_c, fname_e in [
                ("BHC", "bhc_centroid", "bhc_train_embeddings", "bhc_centroid.npy", "bhc_embeddings.npy"),
                ("DI",  "di_centroid",  "di_train_embeddings",  "di_centroid.npy",  "di_embeddings.npy"),
            ]:
                emb_path = kb_dir / fname_e
                cen_path = kb_dir / fname_c
                if emb_path.exists():
                    setattr(self, attr_e, np.load(emb_path))
                    self.logger.info(f"Loaded {role} training embeddings from file, top-K style scoring enabled")
                elif cen_path.exists():
                    setattr(self, attr_c, np.load(cen_path))
                    self.logger.info(f"Loaded {role} style centroid vector from file (fallback)")
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Implement BaseAgent abstract method process
        
        Args:
            **kwargs: contains the following parameters:
                - bhc_results: Dict[str, Dict] - BHC results from each model {model_name: {content: str, ...}}
                - di_results: Dict[str, Dict] - DI results from each model {model_name: {content: str, ...}}
                - ehr_text: str - original EHR input text
                - shared_context: str - shared context
                - bhc_specific_context: str - BHC-specific context
                - di_specific_context: str - DI-specific context
        
        Returns:
            verification result dict with per-model scores and details
        """
        bhc_results = kwargs.get("bhc_results", {})
        di_results = kwargs.get("di_results", {})
        ehr_text = kwargs.get("ehr_text", "")
        shared_context = kwargs.get("shared_context", "")
        bhc_specific_context = kwargs.get("bhc_specific_context", "")
        di_specific_context = kwargs.get("di_specific_context", "")
        
        # Get all model names
        model_names = set(bhc_results.keys()) | set(di_results.keys())
        
        verification_results = {}
        
        for model_name in model_names:
            bhc_content = bhc_results.get(model_name, {}).get("content", "")
            di_content = di_results.get(model_name, {}).get("content", "")
            
            if not bhc_content and not di_content:
                continue
            
            # Verify results for each model
            model_verification = self._verify_model_output(
                model_name=model_name,
                bhc_content=bhc_content,
                di_content=di_content,
                ehr_text=ehr_text,
                shared_context=shared_context,
                bhc_specific_context=bhc_specific_context,
                di_specific_context=di_specific_context
            )
            
            verification_results[model_name] = model_verification
        
        # Summarize results
        summary = self._summarize_results(verification_results)
        
        return {
            "model_results": verification_results,
            "summary": summary,
            "logs": self.verification_logs
        }
    
    def _verify_model_output(
        self,
        model_name: str,
        bhc_content: str,
        di_content: str,
        ehr_text: str,
        shared_context: str,
        bhc_specific_context: str,
        di_specific_context: str
    ) -> Dict[str, Any]:
        """
        Verify output for a single model
        
        Returns:
            {
                "fact_verification": {...},
                "logic_verification": {...},
                "style_verification": {...},
                "overall_score": float
            }
        """
        log_entry = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "verification_steps": []
        }
        
        # 1. Factual hallucination verification
        fact_result = self._verify_factual_hallucination(
            bhc_content, ehr_text, shared_context, bhc_specific_context
        )
        log_entry["verification_steps"].append({
            "step": "fact_verification",
            "result": fact_result
        })
        
        # 2. Logical hallucination verification
        logic_result = self._verify_logical_hallucination(
            bhc_content, di_content
        )
        log_entry["verification_steps"].append({
            "step": "logic_verification",
            "result": logic_result
        })
        
        # 3. Writing style verification
        style_result = self._verify_writing_style(
            bhc_content, di_content
        )
        log_entry["verification_steps"].append({
            "step": "style_verification",
            "result": style_result
        })
        
        # Compute overall score (weighted average using configured weights)
        overall_score = (
            fact_result["score"] * self.weights["fact"] +
            logic_result["score"] * self.weights["logic"] +
            style_result["score"] * self.weights["style"]
        )
        
        log_entry["overall_score"] = overall_score
        self.verification_logs.append(log_entry)
        
        # Aggregate improvement log
        improvement_log = []
        if "improvement_log" in fact_result:
            improvement_log.extend(fact_result["improvement_log"])
        if "improvement_log" in logic_result:
            improvement_log.extend(logic_result["improvement_log"])
        
        return {
            "fact_verification": fact_result,
            "logic_verification": logic_result,
            "style_verification": style_result,
            "overall_score": overall_score,
            "weights": self.weights,
            "improvement_log": improvement_log
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
        
        Returns:
            Sentence list
        """
        if not text or not text.strip():
            return []
        
        # Split text into sentences using regex (periods, question marks, exclamation marks, etc.)
        sentences = re.split(r'([.!?。！？]\s*)', text)
        
        # Merge delimiters with sentences
        result = []
        current_sentence = ""
        
        for i, part in enumerate(sentences):
            if not part.strip():
                continue
            
            # If delimiter, append to current sentence
            if re.match(r'^[.!?。！？]\s*$', part):
                current_sentence += part
                if current_sentence.strip():
                    result.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Append last sentence if not ending with punctuation
        if current_sentence.strip():
            result.append(current_sentence.strip())
        
        # Filter out sentences that are too short (likely split errors)
        result = [s for s in result if len(s) > 10]
        
        return result
    
    def _verify_factual_hallucination(
        self,
        bhc_content: str,
        ehr_text: str,
        shared_context: str,
        bhc_specific_context: str
    ) -> Dict[str, Any]:
        """
        Factual hallucination check: verify BHC against source input at sentence level
        
        Optimized method:
        1. Extract and encode sentences from generated BHC
        2. Split and encode source text (shared_context + bhc_specific_context)
        3. Retrieve top-k evidence sentences using generated BHC sentences (embedding similarity)
        4. Compute similarity score for each BHC sentence vs retrieved evidence
        5. Aggregate verification results for all sentences
        """
        if not bhc_content:
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": "BHC content is empty"},
                "errors": ["BHC content is empty"],
                "sentence_results": [],
                "improvement_log": []
            }
        
        if not self.embedding_model:
            return {
                "score": 1.0,
                "passed": True,
                "details": {"message": "Embedding model not loaded, skipping verification"},
                "errors": [],
                "sentence_results": [],
                "improvement_log": []
            }
        
        try:
            # Build evidence text from shared context and BHC-specific context
            evidence_text = f"{shared_context}\n\n{bhc_specific_context}".strip()
            if not evidence_text:
                evidence_text = ehr_text  # If no segmented context, use full EHR text
            
            # Step 1: split BHC into sentences
            bhc_sentences = self._split_into_sentences(bhc_content)
            
            if not bhc_sentences:
                return {
                    "score": 0.0,
                    "passed": False,
                    "details": {"error": "Unable to split BHC into sentences"},
                    "errors": ["Unable to split BHC into sentences"],
                    "sentence_results": [],
                    "improvement_log": []
                }
            
            # Step 2: split evidence text into sentences
            evidence_sentences = self._split_into_sentences(evidence_text)
            
            if not evidence_sentences:
                return {
                    "score": 0.0,
                    "passed": False,
                    "details": {"error": "Unable to split evidence text into sentences"},
                    "errors": ["Unable to split evidence text into sentences"],
                    "sentence_results": [],
                    "improvement_log": []
                }
            
            # Step 3: encode all evidence sentences (batched for efficiency)
            evidence_embeddings = self.embedding_model.encode(evidence_sentences, show_progress_bar=False)
            evidence_embeddings = np.array(evidence_embeddings)
            
            # Step 4: verify each BHC sentence
            sentence_scores = []
            sentence_results = []
            improvement_log = []
            top_k = 3  # retrieve top-k evidence sentences
            
            for idx, bhc_sentence in enumerate(bhc_sentences):
                try:
                    # Encode BHC sentence
                    bhc_sentence_embedding = self.embedding_model.encode(bhc_sentence)
                    bhc_sentence_embedding = np.array(bhc_sentence_embedding)
                    
                    # Step 5: retrieve top-k evidence sentences (cosine similarity)
                    # Compute similarity against all evidence sentences
                    similarities = np.dot(evidence_embeddings, bhc_sentence_embedding) / (
                        np.linalg.norm(evidence_embeddings, axis=1) * np.linalg.norm(bhc_sentence_embedding)
                    )
                    
                    # Get top-k most similar evidence sentences
                    top_k_indices = np.argsort(similarities)[::-1][:top_k]
                    top_k_similarities = similarities[top_k_indices]
                    top_k_evidence_sentences = [evidence_sentences[i] for i in top_k_indices]
                    
                    # Compute mean similarity score across top-k evidence sentences
                    avg_similarity = np.mean(top_k_similarities)
                    normalized_score = (avg_similarity + 1) / 2  # normalize to 0-1
                    
                    sentence_scores.append(normalized_score)
                    
                    # Determine whether sentence passes
                    sentence_passed = normalized_score >= self.thresholds["fact"]
                    
                    # Save retrieved evidence sentence details
                    retrieved_evidence_details = []
                    for ev_idx, ev_sentence, sim in zip(top_k_indices, top_k_evidence_sentences, top_k_similarities):
                        retrieved_evidence_details.append({
                            "evidence_index": int(ev_idx),
                            "evidence_sentence": ev_sentence[:150] + "..." if len(ev_sentence) > 150 else ev_sentence,
                            "similarity": float(sim),
                            "normalized_score": float((sim + 1) / 2)
                        })
                    
                    sentence_result = {
                        "sentence_index": idx,
                        "sentence": bhc_sentence[:100] + "..." if len(bhc_sentence) > 100 else bhc_sentence,
                        "score": float(normalized_score),
                        "passed": sentence_passed,
                        "avg_similarity": float(avg_similarity),
                        "top_k": top_k,
                        "retrieved_evidence_count": len(retrieved_evidence_details),
                        "retrieved_evidence_details": retrieved_evidence_details
                    }
                    sentence_results.append(sentence_result)
                    
                    # If sentence fails, add to improvement log
                    if not sentence_passed:
                        improvement_log.append({
                            "type": "factual_hallucination",
                            "sentence_index": idx,
                            "sentence": bhc_sentence,
                            "score": float(normalized_score),
                            "threshold": self.thresholds["fact"],
                            "top_k_evidence": top_k_evidence_sentences[:2],  # save only first 2 as examples
                            "suggestion": f"This sentence has low consistency with the input evidence ({normalized_score:.3f} < {self.thresholds['fact']}), possible factual hallucination. Check whether the medical description matches the original EHR data."
                        })
                
                except Exception as e:
                    self.logger.warning(f"Error verifying sentence {idx}: {str(e)}")
                    sentence_scores.append(0.0)
                    sentence_results.append({
                        "sentence_index": idx,
                        "sentence": bhc_sentence[:100] + "..." if len(bhc_sentence) > 100 else bhc_sentence,
                        "score": 0.0,
                        "passed": False,
                        "error": str(e)
                    })
            
            # Compute overall score (mean of all sentence scores)
            if sentence_scores:
                fact_score = sum(sentence_scores) / len(sentence_scores)
            else:
                fact_score = 0.0
            
            # Determine whether verification passes
            passed = fact_score >= self.thresholds["fact"]
            
            errors = []
            if not passed:
                failed_count = sum(1 for r in sentence_results if not r.get("passed", False))
                errors.append(f"Factual consistency score too low: {fact_score:.3f} < {self.thresholds['fact']} ({failed_count}/{len(sentence_results)} sentences failed)")
            
            return {
                "score": fact_score,
                "passed": passed,
                "details": {
                    "total_sentences": len(bhc_sentences),
                    "total_evidence_sentences": len(evidence_sentences),
                    "passed_sentences": sum(1 for r in sentence_results if r.get("passed", False)),
                    "failed_sentences": sum(1 for r in sentence_results if not r.get("passed", False)),
                    "avg_sentence_score": float(np.mean(sentence_scores)) if sentence_scores else 0.0,
                    "top_k": top_k
                },
                "errors": errors,
                "sentence_results": sentence_results,
                "improvement_log": improvement_log
            }
            
        except Exception as e:
            self.logger.error(f"Factual hallucination verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
                "sentence_results": [],
                "improvement_log": []
            }
    
    def _verify_logical_hallucination(
        self,
        bhc_content: str,
        di_content: str
    ) -> Dict[str, Any]:
        """
        Logical hallucination check: use QA pairs to verify BHC-DI consistency at sentence-pair level
        
        Method:
        1. Split BHC and DI into sentences
        2. Use BHC sentence to retrieve related QA pairs from the QA dataset
        3. Check logical consistency for each BHC-DI sentence pair
        4. Aggregate verification results for all sentence pairs
        """
        if not bhc_content or not di_content:
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": "BHC or DI content is empty"},
                "errors": ["BHC or DI content is empty"],
                "sentence_pair_results": [],
                "improvement_log": []
            }
        
        try:
            if not self.knowledge_base or not self.knowledge_base.qa_pairs:
                return {
                    "score": 1.0,  # If no QA pairs, pass by default
                    "passed": True,
                    "details": {"message": "QA pair dataset not loaded, skipping logic verification"},
                    "errors": [],
                    "sentence_pair_results": [],
                    "improvement_log": []
                }
            
            if not self.embedding_model:
                return {
                    "score": 1.0,
                    "passed": True,
                    "details": {"message": "Embedding model not loaded, skipping logic verification"},
                    "errors": [],
                    "sentence_pair_results": [],
                    "improvement_log": []
                }
            
            # Split BHC and DI into sentences
            bhc_sentences = self._split_into_sentences(bhc_content)
            di_sentences = self._split_into_sentences(di_content)
            
            if not bhc_sentences or not di_sentences:
                return {
                    "score": 0.0,
                    "passed": False,
                    "details": {"error": "Unable to split BHC or DI into sentences"},
                    "errors": ["Unable to split BHC or DI into sentences"],
                    "sentence_pair_results": [],
                    "improvement_log": []
                }
            
            # For each BHC sentence, retrieve related QA pairs and check corresponding DI sentence
            sentence_pair_scores = []
            sentence_pair_results = []
            improvement_log = []
            
            # Retrieve related QA pairs for each BHC sentence
            for bhc_idx, bhc_sentence in enumerate(bhc_sentences):
                # Use BHC sentence to retrieve related QA pairs from the QA dataset
                retrieved_qa_pairs = self.knowledge_base.rag_retrieve_qa(
                    query=bhc_sentence,
                    top_k=3,  # retrieve 3 QA pairs per sentence
                    similarity_threshold=0.6
                )
                
                if not retrieved_qa_pairs:
                    # If no QA pairs retrieved, assign medium score
                    sentence_pair_scores.append(0.5)
                    sentence_pair_results.append({
                        "bhc_sentence_index": bhc_idx,
                        "bhc_sentence": bhc_sentence[:100] + "..." if len(bhc_sentence) > 100 else bhc_sentence,
                        "score": 0.5,
                        "passed": False,
                        "warning": "No relevant QA pairs found",
                        "retrieved_qa_pairs": 0,
                        "retrieved_qa_details": []  # empty list
                    })
                    continue
                
                # Find most relevant DI sentence via embedding similarity
                best_di_match = None
                best_di_score = 0.0
                best_di_idx = -1
                
                for di_idx, di_sentence in enumerate(di_sentences):
                    # Compute similarity between DI sentence and retrieved answers
                    di_embedding = self.embedding_model.encode(di_sentence)
                    
                    # Compute mean similarity against all retrieved answers
                    answer_similarities = []
                    for question, answer, qa_similarity in retrieved_qa_pairs:
                        answer_embedding = self.embedding_model.encode(answer)
                        answer_similarity = np.dot(di_embedding, answer_embedding) / (
                            np.linalg.norm(di_embedding) * np.linalg.norm(answer_embedding)
                        )
                        # Weighted similarity (factoring in retrieval score)
                        weighted_similarity = answer_similarity * qa_similarity
                        answer_similarities.append(weighted_similarity)
                    
                    avg_similarity = sum(answer_similarities) / len(answer_similarities) if answer_similarities else 0.0
                    normalized_score = (avg_similarity + 1) / 2
                    
                    if normalized_score > best_di_score:
                        best_di_score = normalized_score
                        best_di_match = di_sentence
                        best_di_idx = di_idx
                
                # Record sentence-pair result
                sentence_pair_passed = best_di_score >= self.thresholds["logic"]
                sentence_pair_scores.append(best_di_score)
                
                # Save retrieved QA pair details
                retrieved_qa_details = []
                for q, a, sim in retrieved_qa_pairs:
                    retrieved_qa_details.append({
                        "question": q,
                        "answer": a,
                        "similarity": float(sim)
                    })
                
                sentence_pair_result = {
                    "bhc_sentence_index": bhc_idx,
                    "bhc_sentence": bhc_sentence[:100] + "..." if len(bhc_sentence) > 100 else bhc_sentence,
                    "di_sentence_index": best_di_idx,
                    "di_sentence": best_di_match[:100] + "..." if best_di_match and len(best_di_match) > 100 else (best_di_match or ""),
                    "score": float(best_di_score),
                    "passed": sentence_pair_passed,
                    "retrieved_qa_pairs": len(retrieved_qa_pairs),
                    "retrieved_qa_details": retrieved_qa_details  # save detailed QA pair info
                }
                sentence_pair_results.append(sentence_pair_result)
                
                # If sentence pair fails, add to improvement log
                if not sentence_pair_passed:
                    improvement_log.append({
                        "type": "logical_hallucination",
                        "bhc_sentence_index": bhc_idx,
                        "bhc_sentence": bhc_sentence,
                        "di_sentence_index": best_di_idx,
                        "di_sentence": best_di_match or "",
                        "score": float(best_di_score),
                        "threshold": self.thresholds["logic"],
                            "suggestion": f"This BHC-DI sentence pair has low logical consistency ({best_di_score:.3f} < {self.thresholds['logic']}), possible logical hallucination. Check whether the discharge instructions in DI are logically consistent with the clinical course in BHC."
                    })
            
            # Compute overall score (mean of all sentence-pair scores)
            if sentence_pair_scores:
                logic_score = sum(sentence_pair_scores) / len(sentence_pair_scores)
            else:
                logic_score = 0.0
            
            # Determine whether verification passes
            passed = logic_score >= self.thresholds["logic"]
            
            errors = []
            if not passed:
                failed_count = sum(1 for r in sentence_pair_results if not r.get("passed", False))
                errors.append(f"Logical consistency score too low: {logic_score:.3f} < {self.thresholds['logic']} ({failed_count}/{len(sentence_pair_results)} sentence pairs failed)")
            
            return {
                "score": logic_score,
                "passed": passed,
                "details": {
                    "total_bhc_sentences": len(bhc_sentences),
                    "total_di_sentences": len(di_sentences),
                    "total_sentence_pairs": len(sentence_pair_results),
                    "passed_sentence_pairs": sum(1 for r in sentence_pair_results if r.get("passed", False)),
                    "failed_sentence_pairs": sum(1 for r in sentence_pair_results if not r.get("passed", False)),
                    "avg_sentence_pair_score": float(np.mean(sentence_pair_scores)) if sentence_pair_scores else 0.0
                },
                "errors": errors,
                "sentence_pair_results": sentence_pair_results,
                "improvement_log": improvement_log
            }
            
        except Exception as e:
            self.logger.error(f"Logical hallucination verification failed: {str(e)}")
            return {
                "score": 0.0,
                "passed": False,
                "details": {"error": str(e)},
                "errors": [str(e)],
                "sentence_pair_results": [],
                "improvement_log": []
            }
    
    def _verify_writing_style(
        self,
        bhc_content: str,
        di_content: str
    ) -> Dict[str, Any]:
        """
        Writing style verification.
        Prefer top-K nearest-neighbor scoring (needs train embeddings); fall back to centroid cosine similarity.
        Both methods normalized to [0, 1].
        """
        if not self.embedding_model:
            return {
                "score": 1.0, "passed": True,
                "details": {"message": "Embedding model not loaded, skipping style verification"},
                "errors": []
            }

        def _centroid_score(content: str, centroid: np.ndarray, label: str):
            emb = self.embedding_model.encode(content)
            sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9)
            score = (float(sim) + 1) / 2
            return score, {"method": "centroid", "cosine_similarity": float(sim), "normalized_score": score}

        try:
            style_scores  = {}
            style_details = {}

            # ── BHC style scoring ─────────────────────────────────────────────────
            if bhc_content:
                if self.bhc_train_embeddings is not None:
                    sc, det = self._compute_topk_style_score(
                        bhc_content, self.bhc_train_embeddings, role="BHC"
                    )
                elif self.bhc_centroid is not None:
                    sc, det = _centroid_score(bhc_content, self.bhc_centroid, "BHC")
                else:
                    sc, det = 0.0, {"error": "BHC style reference vector not loaded"}
            else:
                sc, det = 0.0, {"error": "BHC content is empty"}
            style_scores["bhc"]  = sc
            style_details["bhc"] = det

            # ── DI style scoring ──────────────────────────────────────────────────
            if di_content:
                if self.di_train_embeddings is not None:
                    sc, det = self._compute_topk_style_score(
                        di_content, self.di_train_embeddings, role="DI"
                    )
                elif self.di_centroid is not None:
                    sc, det = _centroid_score(di_content, self.di_centroid, "DI")
                else:
                    sc, det = 0.0, {"error": "DI style reference vector not loaded"}
            else:
                sc, det = 0.0, {"error": "DI content is empty"}
            style_scores["di"]  = sc
            style_details["di"] = det

            # ── Overall score (BHC 60% + DI 40%, weights adjustable) ──────────────────────
            bhc_s = style_scores["bhc"]
            di_s  = style_scores["di"]
            if bhc_s > 0 and di_s > 0:
                overall_style_score = bhc_s * 0.6 + di_s * 0.4
            elif bhc_s > 0:
                overall_style_score = bhc_s
            elif di_s > 0:
                overall_style_score = di_s
            else:
                overall_style_score = 0.0

            passed = overall_style_score >= self.thresholds["style"]
            errors = []
            if not passed:
                errors.append(f"Writing style score too low: {overall_style_score:.3f} < {self.thresholds['style']}")

            return {
                "score": overall_style_score,
                "passed": passed,
                "details": style_details,
                "errors": errors
            }

        except Exception as e:
            self.logger.error(f"Writing style verification failed: {str(e)}")
            return {"score": 0.0, "passed": False, "details": {"error": str(e)}, "errors": [str(e)]}
    
    def _compute_topk_style_score(
        self,
        content: str,
        train_embeddings: np.ndarray,
        top_k: int = 10,
        sample_size: int = 1000,
        role: str = "TEXT"
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Top-K nearest-neighbor style score: mean similarity to K most similar training samples.
        More robust than single centroid; avoids high-dimensional mean compression.
        """
        try:
            content_embedding = self.embedding_model.encode(content)

            if len(train_embeddings) > sample_size:
                sample_indices = np.random.choice(len(train_embeddings), size=sample_size, replace=False)
                sampled_embeddings = train_embeddings[sample_indices]
            else:
                sampled_embeddings = train_embeddings

            content_norm = np.linalg.norm(content_embedding)
            if content_norm == 0:
                return 0.0, {"error": "Generated text embedding is a zero vector"}

            embeddings_norm = np.linalg.norm(sampled_embeddings, axis=1)
            embeddings_norm[embeddings_norm == 0] = 1.0

            cosine_similarities = np.dot(sampled_embeddings, content_embedding) / (
                embeddings_norm * content_norm
            )

            top_k_actual = min(top_k, len(cosine_similarities))
            top_k_indices = np.argpartition(cosine_similarities, -top_k_actual)[-top_k_actual:]
            top_k_similarities = cosine_similarities[top_k_indices]

            mean_sim = float(np.mean(top_k_similarities))
            max_sim  = float(np.max(top_k_similarities))
            med_sim  = float(np.median(top_k_similarities))
            style_score = (mean_sim + 1) / 2

            self.logger.debug(
                f"{role} style verification (top-{top_k_actual}) — mean: {mean_sim:.3f}, max: {max_sim:.3f}, median: {med_sim:.3f}"
            )
            return style_score, {
                "method": "topk_nearest_neighbors",
                "top_k": top_k_actual,
                "sampled_from": len(sampled_embeddings),
                "mean_similarity": mean_sim,
                "max_similarity": max_sim,
                "median_similarity": med_sim,
                "normalized_score": float(style_score),
            }
        except Exception as e:
            self.logger.error(f"Top-K style verification failed: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def _summarize_results(self, verification_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize verification results
        
        Returns:
            {
                "model_scores": {model_name: overall_score},
                "best_model": str,
                "average_score": float,
                "statistics": {...}
            }
        """
        if not verification_results:
            return {
                "model_scores": {},
                "best_model": None,
                "average_score": 0.0,
                "statistics": {}
            }
        
        model_scores = {
            model_name: result["overall_score"]
            for model_name, result in verification_results.items()
        }
        
        # Find best model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0] if model_scores else None
        
        # Compute average score
        average_score = sum(model_scores.values()) / len(model_scores) if model_scores else 0.0
        
        # Statistics
        fact_scores = [result["fact_verification"]["score"] for result in verification_results.values()]
        logic_scores = [result["logic_verification"]["score"] for result in verification_results.values()]
        style_scores = [result["style_verification"]["score"] for result in verification_results.values()]
        
        statistics = {
            "num_models": len(verification_results),
            "fact_scores": {
                "mean": float(np.mean(fact_scores)) if fact_scores else 0.0,
                "std": float(np.std(fact_scores)) if fact_scores else 0.0,
                "min": float(np.min(fact_scores)) if fact_scores else 0.0,
                "max": float(np.max(fact_scores)) if fact_scores else 0.0
            },
            "logic_scores": {
                "mean": float(np.mean(logic_scores)) if logic_scores else 0.0,
                "std": float(np.std(logic_scores)) if logic_scores else 0.0,
                "min": float(np.min(logic_scores)) if logic_scores else 0.0,
                "max": float(np.max(logic_scores)) if logic_scores else 0.0
            },
            "style_scores": {
                "mean": float(np.mean(style_scores)) if style_scores else 0.0,
                "std": float(np.std(style_scores)) if style_scores else 0.0,
                "min": float(np.min(style_scores)) if style_scores else 0.0,
                "max": float(np.max(style_scores)) if style_scores else 0.0
            }
        }
        
        return {
            "model_scores": model_scores,
            "best_model": best_model,
            "average_score": average_score,
            "statistics": statistics
        }
    
    def save_verification_logs(self, output_dir: Path, case_name: str):
        """
        Save verification logs to file
        
        Args:
            output_dir: Output directory
            case_name: Case name
        """
        log_file = output_dir / f"{case_name}_verification_log.json"
        
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
        
        with open(log_file, 'w', encoding='utf-8') as f:
            serializable_logs = convert_to_serializable(self.verification_logs)
            json.dump({
                "case_name": case_name,
                "verification_logs": serializable_logs,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Verification log saved: {log_file}")

