"""
Knowledge base construction and retrieval system
Extracts knowledge from the training set for retrieval during verification
"""
import json
import pickle
import sys
import gc
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

# Add project root to path to ensure utils module can be imported
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss is not installed, falling back to linear search (slower)")
    print("Install with: pip install faiss-cpu")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base: stores and retrieves training-set knowledge.

    Features:
    1. Fact check: RAG corpus storing training BHC/DI embeddings and text
    2. Logic check: extract BHC->DI QA pairs from training data for alignment verification
    3. Style verification: compute BHC/DI style centroids for similarity scoring
    """
    
    def __init__(self, kb_dir: Path = Path("outputs/knowledge_base")):
        """
        Args:
            kb_dir: Knowledge base storage directory
        """
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG retrieval dataset (for fact-checking)
        self.bhc_embeddings = None  # BHC embedding matrix
        self.bhc_texts = []  # Corresponding BHC texts (after chunking)
        self.bhc_text_metadata = []  # BHC text metadata [(original_idx, chunk_idx, start_pos, end_pos), ...]
        self.bhc_centroid = None  # BHC style centroid vector (for style verification)
        self.bhc_index = None  # FAISS index (for fast retrieval)
        
        self.di_embeddings = None  # DI embedding matrix
        self.di_texts = []  # Corresponding DI texts (after chunking)
        self.di_text_metadata = []  # DI text metadata [(original_idx, chunk_idx, start_pos, end_pos), ...]
        self.di_centroid = None  # DI style centroid vector (for style verification)
        self.di_index = None  # FAISS index (for fast retrieval)
        
        # QA pairs for logic checking (for logical alignment verification)
        self.qa_pairs = []  # QA pair list [(question, answer, original_idx), ...]
        self.qa_embeddings = None  # QA pair embedding matrix (question+answer concatenated then encoded)
        self.qa_index = None  # QA pair FAISS index
        
        # Embedding cache (to avoid re-encoding)
        self.embedding_cache = {}  # {text_hash: embedding}
        self.cache_file = self.kb_dir / "embedding_cache.pkl"
        
        # Embedding model
        self.embedding_model = None
        if HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # LLM client (for extracting QA pairs)
        self.llm_client = None
        try:
            from utils.llm_client import LLMClient
            self.llm_client = LLMClient()
            logger.info("LLM client loaded")
        except Exception as e:
            logger.warning(f"Failed to load LLM client: {e}")
        
        # Load cache
        self._load_embedding_cache()
    
    def build_from_training_data(
        self,
        train_file: str = "data/discharge_target_train.csv",
        sample_size: Optional[int] = None,
        chunk_texts: bool = True,
        chunk_size: int = 200,  # Number of characters per chunk
        chunk_overlap: int = 50  # Number of overlapping characters between chunks
    ):
        """
        Build knowledge base from training set (RAG corpus + style verification embeddings).

        Args:
            train_file: Training set file path
            sample_size: Sample size (None means use all data)
        """
        print("=" * 80)
        print("Building knowledge base")
        print("=" * 80)
        
        train_path = Path(train_file)
        if not train_path.exists():
            raise FileNotFoundError(f"Training set file not found: {train_path}")
        
        print(f"\n[Step 1/3] Reading training set file")
        print(f"  File path: {train_path}")
        print(f"  Reading...")
        
        df = pd.read_csv(train_path, low_memory=False)
        total_rows = len(df)
        print(f"  ✓ Read complete, {total_rows:,} rows total")
        
        # Check data columns
        required_columns = ['brief_hospital_course', 'discharge_instructions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print(f"  ✓ Column check passed: {', '.join(required_columns)}")
        
        # Check for null values
        bhc_empty = df['brief_hospital_course'].isna().sum()
        di_empty = df['discharge_instructions'].isna().sum()
        print(f"  Data quality check:")
        print(f"    - BHC null values: {bhc_empty:,} ({bhc_empty/total_rows*100:.2f}%)")
        print(f"    - DI null values: {di_empty:,} ({di_empty/total_rows*100:.2f}%)")
        
        if sample_size and len(df) > sample_size:
            print(f"\n  Note: sampling {sample_size:,} samples (from {len(df):,})")
            df = df.sample(n=sample_size, random_state=42)
        else:
            print(f"\n  ✓ Using full dataset: {len(df):,} samples")
        
        print(f"\nProcessing {len(df):,} training samples")
        print()
        
        # Build embedding corpus for BHC and DI (for RAG retrieval and style verification)
        if self.embedding_model:
            print("\n[Step 2/3] Building BHC/DI embedding corpus (fact check + style verification)")
            self._build_style_embeddings(df, chunk_texts=chunk_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Save immediately after step 2 completes
            print("\n  Saving step 2 results (fact check + style verification)...")
            self._save_partial(save_qa=False)  # Only save BHC/DI data, not QA pairs
            print("  ✓ Step 2 results saved")
        else:
            print("\n⚠ Warning: embedding model not loaded, cannot build knowledge base")
            print("  Install with: pip install sentence-transformers")
        
        # Build QA pairs (for logic verification)
        if self.llm_client and self.embedding_model:
            print("\n[Step 3/3] Extracting and encoding QA pairs (logic check)")
            self._build_qa_pairs(df)
        else:
            if not self.llm_client:
                print("\n⚠ Warning: LLM client not loaded, cannot extract QA pairs")
                print("  Please ensure LLM client is configured correctly")
            if not self.embedding_model:
                print("\n⚠ Warning: embedding model not loaded, cannot encode QA pairs")
        
        # Final save (including QA pairs)
        print("\n  Saving complete knowledge base...")
        self.save()
        
        print()
        print("=" * 80)
        print("Knowledge base build complete!")
        print("=" * 80)
        self.print_statistics()
    
    def _build_style_embeddings(
        self, 
        df: pd.DataFrame,
        chunk_texts: bool = True,
        chunk_size: int = 200,
        chunk_overlap: int = 50
    ):
        """
        Build BHC and DI embeddings (with cache).

        Args:
            df: DataFrame
            chunk_texts: Whether to chunk text
            chunk_size: Characters per chunk (if chunk_texts=True)
            chunk_overlap: Overlap characters between chunks (if chunk_texts=True)
        """
        # Read and filter texts (use generator expression to reduce memory usage)
        bhc_texts_raw = [t for t in df['brief_hospital_course'].fillna('').astype(str) if t.strip()]
        di_texts_raw = [t for t in df['discharge_instructions'].fillna('').astype(str) if t.strip()]
        
        # Release DataFrame memory (if no longer needed)
        del df
        gc.collect()
        
        print(f"  Raw BHC text count: {len(bhc_texts_raw):,}")
        print(f"  Raw DI text count: {len(di_texts_raw):,}")
        
        # Check text lengths to decide whether chunking is needed
        avg_bhc_len = np.mean([len(t) for t in bhc_texts_raw[:1000]]) if len(bhc_texts_raw) > 0 else 0
        avg_di_len = np.mean([len(t) for t in di_texts_raw[:1000]]) if len(di_texts_raw) > 0 else 0
        
        print(f"  Text length stats (first 1000 samples):")
        print(f"    - BHC average length: {avg_bhc_len:.0f} chars")
        print(f"    - DI average length: {avg_di_len:.0f} chars")
        
        # If average length is less than 2x chunk_size, recommend skipping chunking
        if chunk_texts and (avg_bhc_len < chunk_size * 2 and avg_di_len < chunk_size * 2):
            print(f"\n  💡 Tip: average text length is short, consider skipping chunking for speed")
            print(f"     Use --no-chunk to skip chunking")
        
        # Chunk processing (if needed)
        if chunk_texts:
            print(f"\n  Text chunking settings: chunk_size={chunk_size}, overlap={chunk_overlap}")
            # Dynamically adjust batch size based on text length
            if avg_bhc_len < 500 and avg_di_len < 500:
                processing_batch_size = 32  # larger batch for short texts
            elif avg_bhc_len < 1000 and avg_di_len < 1000:
                processing_batch_size = 16  # medium length
            else:
                processing_batch_size = 8   # Use smaller batches for long texts
            
            print(f"  Using streaming mode (chunk+encode per batch, {processing_batch_size:,} texts/batch)")
            
            # Streaming BHC: chunk a batch -> encode a batch -> accumulate results
            self.bhc_texts, self.bhc_text_metadata, self.bhc_embeddings = self._chunk_and_encode_streaming(
                bhc_texts_raw, chunk_size, chunk_overlap, processing_batch_size, "BHC"
            )
            print(f"  ✓ BHC processing complete: {len(self.bhc_texts):,} chunks, embedding shape {self.bhc_embeddings.shape}")
            
            # Streaming DI: chunk a batch -> encode a batch -> accumulate results
            self.di_texts, self.di_text_metadata, self.di_embeddings = self._chunk_and_encode_streaming(
                di_texts_raw, chunk_size, chunk_overlap, processing_batch_size, "DI"
            )
            print(f"  ✓ DI processing complete: {len(self.di_texts):,} chunks, embedding shape {self.di_embeddings.shape}")
        else:
            self.bhc_texts = bhc_texts_raw
            self.di_texts = di_texts_raw
            # Create metadata (each text as one chunk)
            self.bhc_text_metadata = [(i, 0, 0, len(text)) for i, text in enumerate(bhc_texts_raw)]
            self.di_text_metadata = [(i, 0, 0, len(text)) for i, text in enumerate(di_texts_raw)]
            print(f"\n  BHC text count: {len(self.bhc_texts):,} (no chunking)")
            print(f"  DI text count: {len(self.di_texts):,} (no chunking)")
            
            # Dynamically adjust encoding batch size based on text count
            if len(self.bhc_texts) > 10000:
                encode_batch_size = 64  # larger batch for large text volumes
            elif len(self.bhc_texts) > 5000:
                encode_batch_size = 32
            else:
                encode_batch_size = 16
            
            # Encode BHC
            if self.bhc_texts:
                print(f"\n  Encoding BHC embeddings (batch size: {encode_batch_size})...")
                self.bhc_embeddings = self._encode_with_cache(
                    self.bhc_texts,
                    show_progress_bar=True,
                    batch_size=encode_batch_size
                )
                print(f"  ✓ BHC embedding complete: {self.bhc_embeddings.shape}")
            
            # Encode DI
            if self.di_texts:
                print(f"\n  Encoding DI embeddings (batch size: {encode_batch_size})...")
                self.di_embeddings = self._encode_with_cache(
                    self.di_texts,
                    show_progress_bar=True,
                    batch_size=encode_batch_size
                )
                print(f"  ✓ DI embedding complete: {self.di_embeddings.shape}")
        
        # Build FAISS index and compute centroid (BHC)
        if self.bhc_embeddings is not None and len(self.bhc_embeddings) > 0:
            if HAS_FAISS:
                print(f"\n  Building BHC FAISS index...")
                self.bhc_index = self._build_faiss_index(self.bhc_embeddings)
                print(f"  ✓ BHC FAISS index built")
            
            # Compute BHC style centroid (for style verification)
            print(f"  Computing BHC style centroid...")
            self.bhc_centroid = np.mean(self.bhc_embeddings, axis=0)
            print(f"  ✓ BHC style centroid: {self.bhc_centroid.shape}")
        
        # Build FAISS index and compute centroid (DI)
        if self.di_embeddings is not None and len(self.di_embeddings) > 0:
            if HAS_FAISS:
                print(f"\n  Building DI FAISS index...")
                self.di_index = self._build_faiss_index(self.di_embeddings)
                print(f"  ✓ DI FAISS index built")
            
            # Compute DI style centroid (for style verification)
            print(f"  Computing DI style centroid...")
            self.di_centroid = np.mean(self.di_embeddings, axis=0)
            print(f"  ✓ DI style centroid: {self.di_centroid.shape}")
        
        # Save cache
        self._save_embedding_cache()
    
    def _build_qa_pairs(self, df: pd.DataFrame):
        """
        Extract BHC->DI QA pairs from training data (logic check).
        Save incrementally to avoid data loss on failure.

        Args:
            df: DataFrame with brief_hospital_course and discharge_instructions columns
        """
        print(f"  Extracting QA pairs from training data...")
        
        # Try to load existing QA pairs (supports resume)
        existing_qa_pairs = self._load_qa_pairs()
        processed_indices = set(idx for _, _, idx in existing_qa_pairs) if existing_qa_pairs else set()
        
        if existing_qa_pairs:
            print(f"  Found existing QA pairs: {len(existing_qa_pairs):,}, will continue extracting...")
            self.qa_pairs = existing_qa_pairs
        else:
            self.qa_pairs = []
        
        # Filter valid data
        valid_data = df[
            df['brief_hospital_course'].notna() & 
            df['discharge_instructions'].notna()
        ].copy()
        
        valid_data['brief_hospital_course'] = valid_data['brief_hospital_course'].astype(str)
        valid_data['discharge_instructions'] = valid_data['discharge_instructions'].astype(str)
        
        # Filter empty texts
        valid_data = valid_data[
            (valid_data['brief_hospital_course'].str.strip() != '') &
            (valid_data['discharge_instructions'].str.strip() != '')
        ]
        
        # Filter already-processed data
        if processed_indices:
            valid_data = valid_data[~valid_data.index.isin(processed_indices)]
        
        print(f"  Valid data pairs: {len(valid_data):,} (already processed: {len(processed_indices):,})")
        
        if len(valid_data) == 0:
            print("  ✓ All data has been processed")
            if self.qa_pairs:
                # Encode QA pairs if not yet encoded
                if self.qa_embeddings is None:
                    self._encode_qa_pairs()
            return
        
        # Batch processing to extract QA pairs (avoid excessive API calls)
        qa_batch_size = 10  # Process 10 per batch
        total_batches = (len(valid_data) + qa_batch_size - 1) // qa_batch_size
        
        # Use tqdm for progress display (if available)
        if HAS_TQDM and total_batches > 1:
            batch_iterator = tqdm(
                range(total_batches),
                desc="    Extracting QA pairs",
                unit="batch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            batch_iterator = range(total_batches)
        
        save_interval = 5  # Save every 5 batches
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * qa_batch_size
            end_idx = min(start_idx + qa_batch_size, len(valid_data))
            batch_data = valid_data.iloc[start_idx:end_idx]
            
            # Extract QA pairs for the current batch
            batch_qa_pairs = []
            for idx, row in batch_data.iterrows():
                bhc = row['brief_hospital_course'].strip()
                di = row['discharge_instructions'].strip()
                original_idx = row.name if hasattr(row, 'name') else idx
                
                # Use LLM to extract QA pairs
                try:
                    qa_pair = self._extract_qa_pair(bhc, di)
                    if qa_pair:
                        batch_qa_pairs.append((qa_pair['question'], qa_pair['answer'], original_idx))
                    else:
                        # If extraction fails, use original text as fallback
                        batch_qa_pairs.append((bhc, di, original_idx))
                except Exception as e:
                    logger.warning(f"Failed to extract QA pair (idx={original_idx}): {e}")
                    # If extraction fails, use original text as fallback
                    batch_qa_pairs.append((bhc, di, original_idx))
            
            # Add to overall list
            self.qa_pairs.extend(batch_qa_pairs)
            
            # Save periodically (save while extracting)
            if (batch_idx + 1) % save_interval == 0 or (batch_idx + 1) == total_batches:
                self._save_qa_pairs_only()
                if HAS_TQDM:
                    batch_iterator.set_postfix(saved=f"{len(self.qa_pairs):,}")
            
            # Perform garbage collection periodically
            if (batch_idx + 1) % 5 == 0:
                gc.collect()
        
        if not HAS_TQDM and total_batches > 1:
            print()  # Newline
        
        print(f"  ✓ Extraction complete, {len(self.qa_pairs):,} QA pairs total")
        
        # Encode QA pairs
        if self.qa_pairs:
            self._encode_qa_pairs()
    
    def _encode_qa_pairs(self):
        """Encode QA pairs"""
        if self.qa_embeddings is not None:
            print(f"  QA pairs already encoded, skipping")
            return
        
        print(f"  Encoding QA pair embeddings...")
        # Concatenate QA pair then encode (question + answer)
        qa_texts = [f"Question: {q}\nAnswer: {a}" for q, a, _ in self.qa_pairs]
        self.qa_embeddings = self._encode_with_cache(
            qa_texts,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"  ✓ QA pair embedding complete: {self.qa_embeddings.shape}")
        
        # Build FAISS index (for fast retrieval)
        if HAS_FAISS:
            print(f"  Building QA pair FAISS index...")
            self.qa_index = self._build_faiss_index(self.qa_embeddings)
            print(f"  ✓ QA pair FAISS index built")
        
        # Save encoded results
        self._save_qa_pairs_only()
    
    def _extract_qa_pair(self, bhc: str, di: str) -> Optional[Dict[str, str]]:
        """
        Use LLM to extract a QA pair from BHC and DI.

        Args:
            bhc: Brief Hospital Course text
            di: Discharge Instructions text

        Returns:
            {"question": str, "answer": str} or None
        """
        if not self.llm_client:
            return None
        
        prompt = f"""Extract a question-answer pair from the following Brief Hospital Course (BHC) and Discharge Instructions (DI).

Requirements:
1. The question MUST be in English and should be DETAILED and SPECIFIC, describing:
   - Specific symptoms, signs, or clinical findings mentioned in the BHC
   - Diagnostic test results, imaging findings, or lab values
   - Treatment procedures, medications, or interventions performed
   - Clinical course or progression of the condition
   - Include multiple relevant details from the BHC to make the question comprehensive

2. The answer MUST be in English and should correspond to relevant discharge instructions from the DI that logically follow from the BHC information

3. The QA pair should demonstrate clear logical connection between the clinical course (BHC) and discharge guidance (DI)

4. The question should be detailed enough to be useful for retrieval-augmented generation - it should contain enough clinical context to match similar cases

Example of a GOOD question:
"What specific symptoms (e.g., chest pain, shortness of breath) did the patient present with, what diagnostic tests were performed (e.g., cardiac catheterization, EKG findings), and what interventions were done (e.g., stent placement, medications administered)?"

Examples of BAD questions (do NOT write these):
- "What happened during the hospital stay?" (too generic — no clinical details)
- "What medications was the patient discharged on?" (too generic — no clinical context linking to the BHC findings)

BHC (Brief Hospital Course):
{bhc}

DI (Discharge Instructions):
{di}

Return ONLY a JSON object in this format:
{{
    "question": "Detailed question in English based on BHC",
    "answer": "Corresponding discharge instruction in English from DI"
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.llm_client.generate(
                model_name="gpt-4o-mini",
                prompt=prompt,
                system_prompt="You are a medical documentation expert specializing in extracting detailed, clinically meaningful question-answer pairs from MIMIC-IV hospital discharge records. Always respond in English. Create questions that are specific, detailed, and contain enough clinical context to be useful for retrieval. The question must contain enough clinical specificity that a retrieval system could match it to similar — but not identical — patient cases.",
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse JSON response
            
            # Try to extract JSON (supports multi-line JSON)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                try:
                    qa_dict = json.loads(json_match.group())
                    if 'question' in qa_dict and 'answer' in qa_dict:
                        question = qa_dict['question'].strip()
                        answer = qa_dict['answer'].strip()
                        
                        # Quality check: filter out QA pairs that are too short or low quality
                        if self._is_valid_qa_pair(question, answer, bhc, di):
                            return {"question": question, "answer": answer}
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, try simple extraction
            lines = response.strip().split('\n')
            question = None
            answer = None
            
            for line in lines:
                if 'question' in line.lower():
                    question = re.sub(r'^.*?["\']?\s*question["\']?\s*[:：]\s*["\']?', '', line, flags=re.IGNORECASE).strip('"\'')
                elif 'answer' in line.lower():
                    answer = re.sub(r'^.*?["\']?\s*answer["\']?\s*[:：]\s*["\']?', '', line, flags=re.IGNORECASE).strip('"\'')
            
            if question and answer:
                question = question.strip()
                answer = answer.strip()
                # Quality check
                if self._is_valid_qa_pair(question, answer, bhc, di):
                    return {"question": question, "answer": answer}
            
            # If all methods fail, return None
            return None
            
        except Exception as e:
            logger.warning(f"LLM failed to extract QA pair: {e}")
            return None
    
    def _is_valid_qa_pair(self, question: str, answer: str, bhc: str, di: str) -> bool:
        """
        Check QA pair quality.

        Args:
            question: Question text
            answer: Answer text
            bhc: Original BHC text
            di: Original DI text

        Returns:
            True if valid, False otherwise
        """
        # Check length (too short may indicate poor quality)
        if len(question) < 30 or len(answer) < 20:
            return False
        
        # Check for Chinese characters (should all be English)
        chinese_chars = re.search(r'[\u4e00-\u9fff]', question + answer)
        if chinese_chars:
            return False
        
        # Check if question is too generic (contains common vague phrases)
        generic_patterns = [
            r'what happened',
            r'what was',
            r'what did',
            r'how was',
            r'what is the',
            r'describe',
        ]
        question_lower = question.lower()
        # If question is too short and contains only vague phrases, consider it low quality
        if len(question) < 50:
            for pattern in generic_patterns:
                if re.search(pattern, question_lower) and len(question.split()) < 10:
                    return False
        
        # Check if question contains enough BHC-related information (at least some keywords)
        bhc_lower = bhc.lower()
        question_lower = question.lower()
        
        # Check if question mentions medical-related terms
        medical_keywords = ['symptom', 'diagnosis', 'test', 'treatment', 'medication', 
                          'procedure', 'finding', 'result', 'patient', 'clinical',
                          'pain', 'fever', 'blood', 'imaging', 'lab', 'examination']
        has_medical_context = any(keyword in question_lower for keyword in medical_keywords)
        
        if not has_medical_context and len(question) < 80:
            return False
        
        return True
    
    def _chunk_and_encode_streaming(
        self,
        texts: List[str],
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        role: str = "text"
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Streaming pipeline: chunk a batch -> encode a batch -> accumulate ->
        free memory -> continue with the next batch.
        Avoids holding all chunks in memory at once.

        Args:
            texts: List of raw texts
            chunk_size: Characters per chunk
            chunk_overlap: Overlap characters between chunks
            batch_size: Number of raw texts per batch
            role: Role label (for progress display)

        Returns:
            (chunked_texts, metadata, embeddings)
        """
        all_chunked_texts = []
        all_metadata = []
        # Use list pre-allocation, but a more efficient approach is to accumulate directly into a numpy array
        embedding_list = []  # Temporary storage to avoid frequent vstack calls
        total_chunks = 0
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Use tqdm for progress display (if available)
        if HAS_TQDM and total_batches > 1:
            batch_iterator = tqdm(
                range(total_batches),
                desc=f"    {role} streaming",
                unit="batch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            batch_iterator = range(total_batches)
        
        # Get embedding dimension (for pre-allocation)
        if self.embedding_model:
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Step 1: Chunk the current batch
            batch_chunked_texts = []
            batch_metadata = []
            
            for local_idx, text in enumerate(batch_texts):
                orig_idx = start_idx + local_idx
                
                if len(text) <= chunk_size:
                    # Text is too short, no chunking needed
                    batch_chunked_texts.append(text)
                    batch_metadata.append((orig_idx, 0, 0, len(text)))
                else:
                    # Chunk processing
                    start = 0
                    chunk_idx = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunk = text[start:end]
                        
                        if end < len(text):
                            # Look backward for period, newline, etc.
                            for punct in ['. ', '.\n', '\n\n', '。', '。\n']:
                                last_punct = chunk.rfind(punct)
                                if last_punct > chunk_size * 0.7:  # Keep at least 70% of content
                                    chunk = chunk[:last_punct + len(punct)]
                                    end = start + len(chunk)
                                    break
                        
                        if chunk.strip():  # Only add non-empty chunks
                            batch_chunked_texts.append(chunk.strip())
                            batch_metadata.append((orig_idx, chunk_idx, start, end))
                            chunk_idx += 1
                        
                        # Move to next chunk (considering overlap)
                        start = end - chunk_overlap
                        if start >= len(text):
                            break
            
            # Step 2: Immediately encode chunks of the current batch
            if batch_chunked_texts:
                batch_embeddings = self._encode_with_cache(
                    batch_chunked_texts,
                    show_progress_bar=False,  # Do not show inner progress bar; outer bar handles it
                    batch_size=32  # Encoding batch size (adjust based on available memory)
                )
                
                # Step 3: Accumulate results
                all_chunked_texts.extend(batch_chunked_texts)
                all_metadata.extend(batch_metadata)
                embedding_list.append(batch_embeddings)
                total_chunks += len(batch_chunked_texts)
                
                # Step 4: Release memory for the current batch
                del batch_texts, batch_chunked_texts, batch_metadata, batch_embeddings
                
                # Perform GC periodically (lower frequency to improve performance)
                if (batch_idx + 1) % 10 == 0:
                    gc.collect()
            
            # Update progress bar info (if using tqdm)
            if HAS_TQDM and total_batches > 1:
                batch_iterator.set_postfix(
                    chunks=f"{total_chunks:,}",
                    mem_mb=f"{total_chunks * embedding_dim * 4 / 1024 / 1024:.1f}"  # Estimated memory (MB)
                )
            elif not HAS_TQDM and total_batches > 1:
                # Fall back to simple print
                print(f"    Processing batch {batch_idx + 1}/{total_batches} ({total_chunks:,} chunks generated)", end='\r')
        
        if not HAS_TQDM and total_batches > 1:
            print()  # Newline
        
        # Merge all embeddings at once to avoid repeated vstack calls
        if embedding_list:
            # If list is large, merge in batches to avoid memory spikes
            if len(embedding_list) > 50:
                # Merge into larger blocks first
                merged_chunks = []
                chunk_size_merge = 20
                for i in range(0, len(embedding_list), chunk_size_merge):
                    chunk = embedding_list[i:i+chunk_size_merge]
                    merged_chunks.append(np.vstack(chunk))
                    del chunk
                final_embeddings = np.vstack(merged_chunks)
                del merged_chunks, embedding_list
            else:
                final_embeddings = np.vstack(embedding_list)
                del embedding_list
        else:
            final_embeddings = np.empty((0, embedding_dim))
        
        # Final garbage collection
        gc.collect()
        
        return all_chunked_texts, all_metadata, final_embeddings
    
    def _chunk_texts_batched(
        self, 
        texts: List[str], 
        chunk_size: int, 
        chunk_overlap: int,
        batch_size: int = 4
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """
        Chunk a text list (batched version, avoids OOM).

        Args:
            texts: List of raw texts
            chunk_size: Characters per chunk
            chunk_overlap: Overlap characters between chunks
            batch_size: Number of texts per batch

        Returns:
            (chunked_texts, metadata)
            metadata: [(original_idx, chunk_idx, start_pos, end_pos), ...]
        """
        chunked_texts = []
        metadata = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Use tqdm for progress display (if available)
        if HAS_TQDM and total_batches > 1:
            batch_iterator = tqdm(
                range(total_batches),
                desc="    Text chunking",
                unit="batch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            batch_iterator = range(total_batches)
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Process current batch
            batch_chunked_texts = []
            batch_metadata = []
            
            for local_idx, text in enumerate(batch_texts):
                orig_idx = start_idx + local_idx
                
                if len(text) <= chunk_size:
                    # Text is too short, no chunking needed
                    batch_chunked_texts.append(text)
                    batch_metadata.append((orig_idx, 0, 0, len(text)))
                else:
                    # Chunk processing
                    start = 0
                    chunk_idx = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunk = text[start:end]
                        
                        if end < len(text):
                            # Look backward for period, newline, etc.
                            for punct in ['. ', '.\n', '\n\n', '。', '。\n']:
                                last_punct = chunk.rfind(punct)
                                if last_punct > chunk_size * 0.7:  # Keep at least 70% of content
                                    chunk = chunk[:last_punct + len(punct)]
                                    end = start + len(chunk)
                                    break
                        
                        if chunk.strip():  # Only add non-empty chunks
                            batch_chunked_texts.append(chunk.strip())
                            batch_metadata.append((orig_idx, chunk_idx, start, end))
                            chunk_idx += 1
                        
                        # Move to next chunk (considering overlap)
                        start = end - chunk_overlap
                        if start >= len(text):
                            break
            
            # Append current batch results to total
            chunked_texts.extend(batch_chunked_texts)
            metadata.extend(batch_metadata)
            
            # Free memory for current batch
            del batch_texts, batch_chunked_texts, batch_metadata
            
            # Update progress bar info (if using tqdm)
            if HAS_TQDM and total_batches > 1:
                batch_iterator.set_postfix(chunks=f"{len(chunked_texts):,}")
            elif not HAS_TQDM and total_batches > 1:
                # Fall back to simple print
                print(f"    Processing batch {batch_idx + 1}/{total_batches} ({len(chunked_texts):,} chunks generated)", end='\r')
        
        if not HAS_TQDM and total_batches > 1:
            print()  # Newline
        
        return chunked_texts, metadata
    
    def _chunk_texts(
        self, 
        texts: List[str], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """
        Chunk a text list (single-batch version, for small datasets).

        Args:
            texts: List of raw texts
            chunk_size: Characters per chunk
            chunk_overlap: Overlap characters between chunks

        Returns:
            (chunked_texts, metadata)
            metadata: [(original_idx, chunk_idx, start_pos, end_pos), ...]
        """
        chunked_texts = []
        metadata = []
        
        for orig_idx, text in enumerate(texts):
            if len(text) <= chunk_size:
                # Text is too short, no chunking needed
                chunked_texts.append(text)
                metadata.append((orig_idx, 0, 0, len(text)))
            else:
                # Chunk processing
                start = 0
                chunk_idx = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunk = text[start:end]
                    
                    if end < len(text):
                        # Look backward for period, newline, etc.
                        for punct in ['. ', '.\n', '\n\n', '。', '。\n']:
                            last_punct = chunk.rfind(punct)
                            if last_punct > chunk_size * 0.7:  # Keep at least 70% of content
                                chunk = chunk[:last_punct + len(punct)]
                                end = start + len(chunk)
                                break
                    
                    if chunk.strip():  # Only add non-empty chunks
                        chunked_texts.append(chunk.strip())
                        metadata.append((orig_idx, chunk_idx, start, end))
                        chunk_idx += 1
                    
                    # Move to next chunk (considering overlap)
                    start = end - chunk_overlap
                    if start >= len(text):
                        break
        
        return chunked_texts, metadata
    
    def _load_embedding_cache(self):
        """Load embedding cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} embedding cache entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Compute text hash (cache key)"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _encode_with_cache(self, texts: List[str], show_progress_bar: bool = False, batch_size: int = 32) -> np.ndarray:
        """
        Encode texts with cache (avoids re-encoding).

        Args:
            texts: List of texts
            show_progress_bar: Whether to show a progress bar
            batch_size: Batch size (default 32; lower reduces memory use)

        Returns:
            Embedding matrix
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        # Check cache (optimized: use pre-allocated list)
        embeddings = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        
        for idx, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                embeddings[idx] = self.embedding_cache[text_hash]
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(idx)
        
        # Encode uncached texts
        if texts_to_encode:
            if show_progress_bar:
                print(f"    Need to encode {len(texts_to_encode):,}/{len(texts):,} texts ({len(texts) - len(texts_to_encode):,} from cache)")
            
            new_embeddings = self.embedding_model.encode(
                texts_to_encode,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
                convert_to_numpy=True  # ensure numpy array output
            )
            
            # Update cache and embeddings list
            for i, (text, embedding) in enumerate(zip(texts_to_encode, new_embeddings)):
                text_hash = self._get_text_hash(text)
                self.embedding_cache[text_hash] = embedding
                embeddings[indices_to_encode[i]] = embedding
            
            # Release temporary variables
            del texts_to_encode, new_embeddings
        
        # Convert to numpy array at once to avoid multiple allocations
        result = np.array(embeddings)
        del embeddings
        return result
    
    def compute_bhc_similarity(self, text: str) -> float:
        """
        Compute similarity between text and the BHC style library (with cache).

        Returns:
            Similarity score (0-1)
        """
        if not self.embedding_model or self.bhc_centroid is None:
            return 0.0
        
        # Use cache for encoding
        text_embedding = self._encode_with_cache([text])[0]
        similarity = np.dot(text_embedding, self.bhc_centroid) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(self.bhc_centroid)
        )
        return float(similarity)
    
    def get_bhc_style_centroid(self) -> Optional[np.ndarray]:
        """
        Get BHC style centroid vector.

        Returns:
            BHC centroid, or None if not loaded
        """
        return self.bhc_centroid
    
    def get_di_style_centroid(self) -> Optional[np.ndarray]:
        """
        Get DI style centroid vector.

        Returns:
            DI centroid, or None if not loaded
        """
        return self.di_centroid
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> Optional[Any]:
        """
        Build a FAISS index for fast similarity search.

        Args:
            embeddings: Embedding matrix (n_samples, embedding_dim)

        Returns:
            FAISS index object
        """
        if not HAS_FAISS:
            return None
        
        try:
            dimension = embeddings.shape[1]
            # Use inner product index (suitable for normalized vectors, equivalent to cosine similarity)
            # If vectors are normalized, inner product = cosine similarity
            index = faiss.IndexFlatIP(dimension)  # Inner Product (suitable for normalized vectors)
            
            # Normalize vectors (so that inner product = cosine similarity)
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # Add vectors to index
            index.add(embeddings_normalized.astype('float32'))
            
            return index
        except Exception as e:
            logger.warning(f"FAISS index build failed: {e}, falling back to linear search")
            return None
    
    def rag_retrieve(
        self,
        query: str,
        role: str = "bhc",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        RAG retrieval: fetch related text from the KB by embedding similarity (fact check).

        Prefer FAISS index (fast); fall back to linear search if unavailable.

        Args:
            query: Query text (generated content or input data)
            role: "bhc" or "di"
            top_k: Return top-k most similar texts
            similarity_threshold: Omit results below this similarity value

        Returns:
            [(text, similarity_score), ...] sorted by similarity
        """
        if not self.embedding_model:
            return []
        
        if role == "bhc":
            embeddings = self.bhc_embeddings
            texts = self.bhc_texts
            index = self.bhc_index
            metadata = self.bhc_text_metadata
        elif role == "di":
            embeddings = self.di_embeddings
            texts = self.di_texts
            index = self.di_index
            metadata = self.di_text_metadata
        else:
            return []
        
        if embeddings is None or len(texts) == 0:
            return []
        
        try:
            # Compute query text embedding (using cache)
            query_embedding = self._encode_with_cache([query])[0]
            
            # Use FAISS index (if available)
            if HAS_FAISS and index is not None:
                return self._rag_retrieve_with_faiss(
                    query_embedding, index, texts, top_k, similarity_threshold, metadata
                )
            else:
                # Fall back to linear search
                return self._rag_retrieve_linear(
                    query_embedding, embeddings, texts, top_k, similarity_threshold, metadata
                )
        
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []
    
    def rag_retrieve_qa(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        RAG QA retrieval: fetch related QA pairs by embedding similarity (logic check).

        Prefer FAISS index (fast); fall back to linear search if unavailable.

        Args:
            query: Query text (usually generated BHC or DI)
            top_k: Return top-k most similar QA pairs
            similarity_threshold: Omit results below this similarity value

        Returns:
            [(question, answer, similarity_score), ...] sorted by similarity
        """
        if not self.embedding_model or self.qa_embeddings is None or len(self.qa_pairs) == 0:
            return []
        
        try:
            # Compute query text embedding (using cache)
            query_embedding = self._encode_with_cache([query])[0]
            
            # Build QA pair text list (for retrieval)
            qa_texts = [f"Question: {q}\nAnswer: {a}" for q, a, _ in self.qa_pairs]
            
            # Use FAISS index (if available)
            if HAS_FAISS and self.qa_index is not None:
                results = self._rag_retrieve_with_faiss(
                    query_embedding, self.qa_index, qa_texts, top_k, similarity_threshold, None
                )
            else:
                # Fall back to linear search
                results = self._rag_retrieve_linear(
                    query_embedding, self.qa_embeddings, qa_texts, top_k, similarity_threshold, None
                )
            
            # Convert text results back to QA pair format
            qa_results = []
            for text, score in results:
                # Extract question and answer from text
                if "Question:" in text and "Answer:" in text:
                    parts = text.split("Answer:")
                    question = parts[0].replace("Question:", "").strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    qa_results.append((question, answer, score))
                else:
                    # If format is incorrect, try looking up in qa_pairs
                    for q, a, _ in self.qa_pairs:
                        if f"Question: {q}\nAnswer: {a}" == text:
                            qa_results.append((q, a, score))
                            break
            
            return qa_results
        
        except Exception as e:
            logger.error(f"QA pair RAG retrieval failed: {e}")
            return []
    
    def _rag_retrieve_with_faiss(
        self,
        query_embedding: np.ndarray,
        index: Any,
        texts: List[str],
        top_k: int,
        similarity_threshold: float,
        metadata: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fast retrieval using a FAISS index.

        Args:
            metadata: Text metadata for dedup (one best chunk per original text)
        """
        try:
            # Normalize query vector
            query_normalized = query_embedding.copy().reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_normalized)
            
            # Search top_k (search more to filter by threshold and deduplicate)
            search_k = min(top_k * 20, len(texts))  # Search more candidates to filter by threshold and deduplicate
            distances, indices = index.search(query_normalized, search_k)
            
            # Filter by similarity threshold and build results
            similarities = []
            seen_orig_idx = set()  # Used for deduplication (keep only the most similar chunk per original text)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(texts) and dist >= similarity_threshold:
                    # If metadata is available, deduplicate
                    if metadata:
                        orig_idx = metadata[idx][0]
                        if orig_idx in seen_orig_idx:
                            continue  # Skip other chunks of the same original text
                        seen_orig_idx.add(orig_idx)
                    
                    similarities.append((texts[idx], float(dist)))
                    
                    # If enough deduplicated results found, exit early
                    if len(similarities) >= top_k:
                        break
            
            # Return top_k
            return similarities[:top_k]
        
        except Exception as e:
            logger.warning(f"FAISS retrieval failed: {e}, falling back to linear search")
            return []
    
    def _rag_retrieve_linear(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        texts: List[str],
        top_k: int,
        similarity_threshold: float,
        metadata: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Linear search (fallback).

        Args:
            metadata: Text metadata for dedup (one best chunk per original text)
        """
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return []
        
        # Vectorized computation of cosine similarity
        embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm[embeddings_norm == 0] = 1  # Avoid division by zero
        
        # Batch compute cosine similarity
        cosine_similarities = np.dot(embeddings, query_embedding) / (embeddings_norm.flatten() * query_norm)
        
        # Filter by threshold and build candidate list
        candidates = []
        for i, similarity in enumerate(cosine_similarities):
            if similarity >= similarity_threshold:
                candidates.append((i, float(similarity)))
        
        # If metadata is available, deduplicate (keep only the most similar chunk per original text)
        if metadata and candidates:
            # Sort by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            seen_orig_idx = set()
            similarities = []
            for idx, similarity in candidates:
                orig_idx = metadata[idx][0]
                if orig_idx not in seen_orig_idx:
                    similarities.append((texts[idx], similarity))
                    seen_orig_idx.add(orig_idx)
                    if len(similarities) >= top_k:
                        break
            return similarities
        else:
            # No metadata, sort directly and return
            similarities = [(texts[i], sim) for i, sim in candidates]
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    def _save_partial(self, save_qa: bool = True):
        """
        Partial KB save (BHC/DI only; QA pairs optional).

        Args:
            save_qa: Whether to save QA pairs
        """
        # Save BHC embedding and texts (RAG retrieval dataset)
        if self.bhc_embeddings is not None:
            np.save(self.kb_dir / "bhc_embeddings.npy", self.bhc_embeddings)
            with open(self.kb_dir / "bhc_texts.json", "w", encoding="utf-8") as f:
                json.dump(self.bhc_texts, f, ensure_ascii=False, indent=2)
            with open(self.kb_dir / "bhc_text_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.bhc_text_metadata, f, ensure_ascii=False, indent=2)
            
            # Save FAISS index (if available)
            if HAS_FAISS and self.bhc_index is not None:
                faiss.write_index(self.bhc_index, str(self.kb_dir / "bhc_index.faiss"))
        
        # Save BHC style centroid vector (style verification)
        if self.bhc_centroid is not None:
            np.save(self.kb_dir / "bhc_centroid.npy", self.bhc_centroid)
        
        # Save DI embedding and texts (RAG retrieval dataset)
        if self.di_embeddings is not None:
            np.save(self.kb_dir / "di_embeddings.npy", self.di_embeddings)
            with open(self.kb_dir / "di_texts.json", "w", encoding="utf-8") as f:
                json.dump(self.di_texts, f, ensure_ascii=False, indent=2)
            with open(self.kb_dir / "di_text_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.di_text_metadata, f, ensure_ascii=False, indent=2)
            
            # Save FAISS index (if available)
            if HAS_FAISS and self.di_index is not None:
                faiss.write_index(self.di_index, str(self.kb_dir / "di_index.faiss"))
        
        # Save DI style centroid vector (style verification)
        if self.di_centroid is not None:
            np.save(self.kb_dir / "di_centroid.npy", self.di_centroid)
        
        # Save QA pairs (if requested)
        if save_qa:
            self._save_qa_pairs_only()
        
        # Save embedding cache
        if self.embedding_cache:
            self._save_embedding_cache()
    
    def _save_qa_pairs_only(self):
        """Save QA pairs only (incremental save during extraction)"""
        if self.qa_pairs:
            with open(self.kb_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
                json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
            
            # If already encoded, also save embedding and index
            if self.qa_embeddings is not None:
                np.save(self.kb_dir / "qa_embeddings.npy", self.qa_embeddings)
                if HAS_FAISS and self.qa_index is not None:
                    faiss.write_index(self.qa_index, str(self.kb_dir / "qa_index.faiss"))
    
    def _load_qa_pairs(self) -> List[Tuple[str, str, int]]:
        """Load existing QA pairs (supports resume)"""
        qa_pairs_path = self.kb_dir / "qa_pairs.json"
        if qa_pairs_path.exists():
            try:
                with open(qa_pairs_path, "r", encoding="utf-8") as f:
                    qa_pairs = json.load(f)
                # Try to load embedding (if it exists)
                qa_emb_path = self.kb_dir / "qa_embeddings.npy"
                if qa_emb_path.exists():
                    self.qa_embeddings = np.load(qa_emb_path)
                    # Try to load index
                    qa_index_path = self.kb_dir / "qa_index.faiss"
                    if HAS_FAISS and qa_index_path.exists():
                        try:
                            self.qa_index = faiss.read_index(str(qa_index_path))
                        except:
                            pass
                return qa_pairs
            except Exception as e:
                logger.warning(f"Failed to load QA pairs: {e}")
        return []
    
    def save(self):
        """Save knowledge base to disk (full save including all data)"""
        print("\nSaving knowledge base...")
        
        # Use partial save method but persist all content
        self._save_partial(save_qa=True)
        
        # Print save info
        if self.bhc_embeddings is not None:
            print(f"  ✓ BHC embedding saved: {self.bhc_embeddings.shape}")
            print(f"  ✓ BHC texts saved: {len(self.bhc_texts):,} entries")
            if HAS_FAISS and self.bhc_index is not None:
                print(f"  ✓ BHC FAISS index saved")
        
        if self.bhc_centroid is not None:
            print(f"  ✓ BHC style centroid saved: {self.bhc_centroid.shape}")
        
        if self.di_embeddings is not None:
            print(f"  ✓ DI embedding saved: {self.di_embeddings.shape}")
            print(f"  ✓ DI texts saved: {len(self.di_texts):,} entries")
            if HAS_FAISS and self.di_index is not None:
                print(f"  ✓ DI FAISS index saved")
        
        if self.di_centroid is not None:
            print(f"  ✓ DI style centroid saved: {self.di_centroid.shape}")
        
        if self.qa_pairs:
            print(f"  ✓ QA pairs saved: {len(self.qa_pairs):,} entries")
            if self.qa_embeddings is not None:
                print(f"  ✓ QA pair embeddings saved: {self.qa_embeddings.shape}")
                if HAS_FAISS and self.qa_index is not None:
                    print(f"  ✓ QA pair FAISS index saved")
        
        if self.embedding_cache:
            print(f"  ✓ Embedding cache saved: {len(self.embedding_cache):,} entries")
        
        print(f"\nKnowledge base saved to: {self.kb_dir}")
    
    def load(self):
        """Load knowledge base from disk (RAG corpus + style embeddings)"""
        print(f"Loading knowledge base: {self.kb_dir}")
        
        # Load BHC embedding and texts (RAG retrieval dataset)
        bhc_emb_path = self.kb_dir / "bhc_embeddings.npy"
        if bhc_emb_path.exists():
            self.bhc_embeddings = np.load(bhc_emb_path)
            with open(self.kb_dir / "bhc_texts.json", "r", encoding="utf-8") as f:
                self.bhc_texts = json.load(f)
            # Try to load metadata (if it exists)
            bhc_meta_path = self.kb_dir / "bhc_text_metadata.json"
            if bhc_meta_path.exists():
                with open(bhc_meta_path, "r", encoding="utf-8") as f:
                    self.bhc_text_metadata = json.load(f)
            else:
                # If no metadata exists, create default metadata (for backward compatibility)
                self.bhc_text_metadata = [(i, 0, 0, len(text)) for i, text in enumerate(self.bhc_texts)]
            print(f"  ✓ BHC embedding loaded: {self.bhc_embeddings.shape}, texts: {len(self.bhc_texts):,}")
            
            # Try to load FAISS index; rebuild if it does not exist
            bhc_index_path = self.kb_dir / "bhc_index.faiss"
            if HAS_FAISS and bhc_index_path.exists():
                try:
                    self.bhc_index = faiss.read_index(str(bhc_index_path))
                    print(f"  ✓ BHC FAISS index loaded")
                except Exception as e:
                    logger.warning(f"Failed to load BHC FAISS index: {e}, rebuilding")
                    self.bhc_index = self._build_faiss_index(self.bhc_embeddings)
            elif HAS_FAISS:
                # If index does not exist, rebuild
                print(f"  Rebuilding BHC FAISS index...")
                self.bhc_index = self._build_faiss_index(self.bhc_embeddings)
                if self.bhc_index:
                    print(f"  ✓ BHC FAISS index built")
        
        # Load BHC style centroid vector (style verification)
        centroid_path = self.kb_dir / "bhc_centroid.npy"
        if centroid_path.exists():
            self.bhc_centroid = np.load(centroid_path)
            print(f"  ✓ BHC style centroid loaded: {self.bhc_centroid.shape}")
        
        # Load DI embedding and texts (RAG retrieval dataset)
        di_emb_path = self.kb_dir / "di_embeddings.npy"
        if di_emb_path.exists():
            self.di_embeddings = np.load(di_emb_path)
            with open(self.kb_dir / "di_texts.json", "r", encoding="utf-8") as f:
                self.di_texts = json.load(f)
            # Try to load metadata (if it exists)
            di_meta_path = self.kb_dir / "di_text_metadata.json"
            if di_meta_path.exists():
                with open(di_meta_path, "r", encoding="utf-8") as f:
                    self.di_text_metadata = json.load(f)
            else:
                # If no metadata exists, create default metadata (for backward compatibility)
                self.di_text_metadata = [(i, 0, 0, len(text)) for i, text in enumerate(self.di_texts)]
            print(f"  ✓ DI embedding loaded: {self.di_embeddings.shape}, texts: {len(self.di_texts):,}")
            
            # Try to load FAISS index; rebuild if it does not exist
            di_index_path = self.kb_dir / "di_index.faiss"
            if HAS_FAISS and di_index_path.exists():
                try:
                    self.di_index = faiss.read_index(str(di_index_path))
                    print(f"  ✓ DI FAISS index loaded")
                except Exception as e:
                    logger.warning(f"Failed to load DI FAISS index: {e}, rebuilding")
                    self.di_index = self._build_faiss_index(self.di_embeddings)
            elif HAS_FAISS:
                # If index does not exist, rebuild
                print(f"  Rebuilding DI FAISS index...")
                self.di_index = self._build_faiss_index(self.di_embeddings)
                if self.di_index:
                    print(f"  ✓ DI FAISS index built")
        
        # Load DI style centroid vector (style verification)
        di_centroid_path = self.kb_dir / "di_centroid.npy"
        if di_centroid_path.exists():
            self.di_centroid = np.load(di_centroid_path)
            print(f"  ✓ DI style centroid loaded: {self.di_centroid.shape}")
        
        # Load QA pairs (logic checking)
        qa_emb_path = self.kb_dir / "qa_embeddings.npy"
        if qa_emb_path.exists():
            self.qa_embeddings = np.load(qa_emb_path)
            with open(self.kb_dir / "qa_pairs.json", "r", encoding="utf-8") as f:
                self.qa_pairs = json.load(f)
            print(f"  ✓ QA pair embeddings loaded: {self.qa_embeddings.shape}, QA pairs: {len(self.qa_pairs):,}")
            
            # Try to load FAISS index; rebuild if it does not exist
            qa_index_path = self.kb_dir / "qa_index.faiss"
            if HAS_FAISS and qa_index_path.exists():
                try:
                    self.qa_index = faiss.read_index(str(qa_index_path))
                    print(f"  ✓ QA pair FAISS index loaded")
                except Exception as e:
                    logger.warning(f"Failed to load QA pair FAISS index: {e}, rebuilding")
                    self.qa_index = self._build_faiss_index(self.qa_embeddings)
            elif HAS_FAISS:
                # If index does not exist, rebuild
                print(f"  Rebuilding QA pair FAISS index...")
                self.qa_index = self._build_faiss_index(self.qa_embeddings)
                if self.qa_index:
                    print(f"  ✓ QA pair FAISS index built")
        
        # Load embedding cache
        self._load_embedding_cache()
        
        print("Knowledge base loaded")
    
    def print_statistics(self):
        """Print knowledge base statistics"""
        print("\nKnowledge base statistics:")
        print(f"  [Fact check] BHC text count: {len(self.bhc_texts):,}")
        print(f"  [Fact check] DI text count: {len(self.di_texts):,}")
        if self.bhc_embeddings is not None:
            print(f"  [Fact check] BHC embedding shape: {self.bhc_embeddings.shape}")
        if self.di_embeddings is not None:
            print(f"  [Fact check] DI embedding shape: {self.di_embeddings.shape}")
        
        if self.qa_pairs:
            print(f"  [Logic check] QA pair count: {len(self.qa_pairs):,}")
        if self.qa_embeddings is not None:
            print(f"  [Logic check] QA pair embedding shape: {self.qa_embeddings.shape}")
        
        if self.bhc_centroid is not None:
            print(f"  [Style check] BHC style centroid shape: {self.bhc_centroid.shape}")
        if self.di_centroid is not None:
            print(f"  [Style check] DI style centroid shape: {self.di_centroid.shape}")
        
        print(f"  Embedding cache entries: {len(self.embedding_cache):,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge base")
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/discharge_target_train.csv",
        help="Training set file path"
    )
    parser.add_argument(
        "--kb-dir",
        type=str,
        default="outputs/knowledge_base",
        help="Knowledge base storage directory"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size (None means use full dataset)"
    )
    parser.add_argument(
        "--chunk-texts",
        action="store_true",
        default=False,  # Default no chunking since each patient's BHC/DI is usually one complete paragraph
        help="Chunk text (default: False; skipping chunks is faster)"
    )
    parser.add_argument(
        "--no-chunk",
        action="store_false",
        dest="chunk_texts",
        help="Do not chunk text (default behavior)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Characters per chunk (default: 200)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap characters between chunks (default: 50)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Knowledge base build parameters")
    print("=" * 80)
    print(f"  Training file: {args.train_file}")
    print(f"  KB directory: {args.kb_dir}")
    print(f"  Sample size: {args.sample_size if args.sample_size else 'full dataset'}")
    print(f"  Text chunking: {args.chunk_texts}")
    if args.chunk_texts:
        print(f"  - chunk_size: {args.chunk_size}")
        print(f"  - chunk_overlap: {args.chunk_overlap}")
    print(f"  Features: RAG retrieval corpus + style verification encoding")
    print("=" * 80 + "\n")
    
    kb = KnowledgeBase(kb_dir=Path(args.kb_dir))
    kb.build_from_training_data(
        train_file=args.train_file,
        sample_size=args.sample_size,  # None means full dataset
        chunk_texts=args.chunk_texts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

# python utils/knowledge_base.py --train-file data/discharge_target_train.csv --kb-dir outputs/knowledge_base
