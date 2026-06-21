"""
Segmentation Agent: splits EHR input into BHC-specific, DI-specific, and shared context using GPT-4O
"""
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
import json
import re
import logging

logger = logging.getLogger(__name__)


class SegmentationAgent(BaseAgent):
    """
    Segmentation Agent: splits full EHR input into:
    - BHC-specific context (Brief Hospital Course)
    - DI-specific context (Discharge Instructions)
    - Shared context (needed by both)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SegmentationAgent", config)
        self.llm_client = LLMClient()
        self.model_name = self.config.get("model_name", "gpt-4o")
        # Read temperature from config (default 0.1 to improve segmentation accuracy)
        self.temperature = self.config.get("temperature", 0.1)
        
        # System prompt (emphasizes preserving original text, categorization only)
        self.system_prompt = """You are a medical information segmentation assistant. Your ONLY task is to COPY and CATEGORIZE sections of the EHR text into three categories. You must NOT summarize, compress, rewrite, or modify ANY text.

CRITICAL RULES:
1. COPY the original text EXACTLY as it appears - word for word, character for character
2. DO NOT use ellipsis (...) or truncate any content
3. DO NOT summarize or paraphrase
4. DO NOT combine multiple sections into one
5. PRESERVE all original formatting, line breaks, spacing, and punctuation
6. If a section belongs to multiple categories, copy it to ALL relevant categories

Your task is to:
1. Read the entire EHR text
2. Identify which sections belong to each category
3. COPY those sections verbatim (without any modification) into the appropriate JSON field
4. If a section appears in the original text, it MUST appear in at least one of the three categories

Categories:

1. **shared_context**: Copy ALL original text sections that are relevant to BOTH Brief Hospital Course (BHC) AND Discharge Instructions (DI). This includes:
   - Patient identification (Stay ID, Subject ID, Hospital Admission ID)
   - Primary diagnoses list
   - Basic demographics (if mentioned in shared sections)
   - Admission/discharge dates (if in shared sections)
   Example text for this category: "Stay ID: 12345. Patient has a history of hypertension and type 2 diabetes. Admitted on ___ and discharged on ___."

2. **bhc_specific_context**: Copy ALL original text sections primarily for Brief Hospital Course. This includes:
   - Emergency Department Stay Information (entire section)
   - Triage Information (entire section)
   - Radiology Reports (entire section, all details)
   - History of Present Illness (entire section)
   - Physical Exam sections (entire sections)
   - Lab Results (entire sections)
   - Imaging and Reports (entire sections)
   - Hospital course details
   - Procedures performed
   - Clinical observations
   - Any other clinical documentation
   Example text for this category: "ED triage note: patient presented with 3/10 chest pain radiating to left arm. HR 98, BP 142/88, O2 sat 97% on room air."

3. **di_specific_context**: Copy ALL original text sections primarily for Discharge Instructions. This includes:
   - Discharge Medications (entire section, all medications)
   - Discharge Instructions (entire section)
   - Follow-up instructions
   - Activity restrictions
   - Diet instructions
   - Warning signs
   - Discharge disposition details
   Example text for this category: "Discharge medications: Metformin 500mg PO BID, Lisinopril 10mg PO daily. Follow up with PCP in 2 weeks."

For borderline sections that could belong to multiple categories, prefer duplicating the text to all relevant categories over omitting it from any.

Return ONLY a valid JSON object with three keys. Each value must contain the COPIED original text sections concatenated together, preserving all original formatting."""
    
    def process(self, ehr_text: str, **kwargs) -> Dict[str, Any]:
        """
        Process EHR text and split into three contexts
        
        Args:
            ehr_text: Full EHR text
            
        Returns:
            {
                "shared_context": str,
                "bhc_specific_context": str,
                "di_specific_context": str,
                "raw_segmentation": str  # raw LLM output
            }
        """
        try:
            self.log(f"Starting rule-based EHR segmentation, length: {len(ehr_text)} chars")
            
            # Use rule-based segmentation method (without calling LLM)
            result = self._rule_based_segmentation(ehr_text)
            
            # Validate the segmentation results
            shared_len = len(result['shared_context'])
            bhc_len = len(result['bhc_specific_context'])
            di_len = len(result['di_specific_context'])
            
            self.log(f"Segmentation complete - shared context: {shared_len} chars, "
                    f"BHC-specific: {bhc_len} chars, "
                    f"DI-specific: {di_len} chars")
            
            # Check if segmentation results are reasonable
            if shared_len < 50:
                self.log("Warning: shared context too short, segmentation may be incomplete", "WARNING")
            if bhc_len < 100:
                self.log("Warning: BHC-specific context too short, segmentation may be incomplete", "WARNING")
            if di_len < 50:
                self.log("Warning: DI-specific context too short, segmentation may be incomplete", "WARNING")
            
            # Calculate total length
            total_extracted = shared_len + bhc_len + di_len
            coverage_ratio = total_extracted / len(ehr_text) if len(ehr_text) > 0 else 0
            self.log(f"Content coverage: {coverage_ratio:.2%} ({total_extracted}/{len(ehr_text)} chars)")
            
            if coverage_ratio < 0.5:
                self.log("Warning: content coverage below 50%, significant information may be missing", "WARNING")
            
            return result
            
        except Exception as e:
            self.log(f"Error during segmentation: {str(e)}", "ERROR")
            raise
    
    def _rule_based_segmentation(self, ehr_text: str) -> Dict[str, Any]:
        """
        Rule- and regex-based segmentation
        Preserve all original text; categorize only, no compression or edits
        """
        shared_parts = []
        bhc_parts = []
        di_parts = []
        
        # Define matching patterns for each section (using regular expressions)
        # Each pattern matches from the start marker to the beginning of the next major section
        
        # 1. Shared context: basic info + final diagnosis list (needed by both BHC and DI)
        # Use a single pattern to capture consecutive ID lines, avoiding overlap from three separate patterns
        shared_patterns = [
            (r'^Stay ID:.*?(?=\n\n)', re.MULTILINE | re.DOTALL),
            (r'^Diagnosis:\s*\n(?:-[^\n]*\n?)+', re.MULTILINE),
            # Discharge Diagnosis contains the final clinical diagnosis list, needed by both BHC and DI
            (r'^Discharge Diagnosis:.*?(?=\n\nDischarge Condition:|\n\nFollowup|\Z)', re.MULTILINE | re.DOTALL),
        ]

        # 2. BHC-specific context: hospital course related information
        bhc_section_patterns = [
            # Emergency Department information
            (r'^Emergency Department Stay Information:.*?(?=\n\nDiagnosis:|\n\nTriage Information:|\n\nRadiology Reports:|\nDischarge Notes|\Z)', re.MULTILINE | re.DOTALL),
            # Triage information
            (r'^Triage Information:.*?(?=\n\nRadiology Reports:|\n\nDischarge|\Z)', re.MULTILINE | re.DOTALL),
            # Radiology Reports (complete section)
            # Fix: no longer using \n\n[A-Z] as stop boundary, since reports internally contain capitalized headers like FINDINGS: / IMPRESSION:
            # Only stop when encountering "Discharge Notes" or end of file
            (r'^Radiology Reports:.*?(?=\nDischarge Notes\s*[\(\[]?|\Z)', re.MULTILINE | re.DOTALL),
        ]

        # Special handling: Discharge Notes section (contains extensive BHC-related information)
        discharge_notes_full = re.search(
            r'Discharge Notes.*?(?=\n\nDischarge Medications:|\Z)',
            ehr_text,
            re.MULTILINE | re.DOTALL
        )
        if discharge_notes_full:
            note_section = discharge_notes_full.group(0).strip()
            if note_section and note_section not in bhc_parts:
                bhc_parts.append(note_section)

        # 3. DI-specific context: discharge-related information
        # Fix: DI patterns no longer use \n\n[A-Z] as stop boundary, replaced with explicit section header boundaries
        di_patterns = [
            # Discharge follow-up/transitional care (contained in Discharge Notes, needs separate extraction to DI)
            (r'^(?:TRANSITIONAL ISSUES|TRANSITIONS OF CARE)[^\n]*\n.*?(?=\n\nMedications on Admission:|\n\nDischarge Medications:|\Z)', re.MULTILINE | re.DOTALL),
            # Discharge medications
            (r'^Discharge Medications:.*?(?=\n\nDischarge Disposition:|\n\nDischarge Diagnosis:|\n\nDischarge Condition:|\n\nFollowup|\Z)', re.MULTILINE | re.DOTALL),
            # Discharge disposition
            (r'^Discharge Disposition:.*?(?=\n\nDischarge Diagnosis:|\n\nDischarge Condition:|\n\nFollowup|\Z)', re.MULTILINE | re.DOTALL),
            # Discharge diagnosis
            (r'^Discharge Diagnosis:.*?(?=\n\nDischarge Condition:|\n\nFollowup|\Z)', re.MULTILINE | re.DOTALL),
            # Discharge condition
            (r'^Discharge Condition:.*?(?=\n\nFollowup|\n\nWEIGHTBEARING|\Z)', re.MULTILINE | re.DOTALL),
            # Weight-bearing status instructions (orthopedic cases)
            (r'^WEIGHTBEARING STATUS:.*?(?=\n\nFollowup|\Z)', re.MULTILINE | re.DOTALL),
            # Follow-up instructions
            (r'^Followup Instructions:.*', re.MULTILINE | re.DOTALL),
        ]
        
        # Extract shared context
        for pattern, flags in shared_patterns:
            matches = list(re.finditer(pattern, ehr_text, flags))
            for match in matches:
                text = match.group(0).strip()
                if text and len(text) > 5:  # Filter out matches that are too short
                    shared_parts.append(text)
        
        # Extract BHC-specific context
        for pattern, flags in bhc_section_patterns:
            matches = list(re.finditer(pattern, ehr_text, flags))
            for match in matches:
                text = match.group(0).strip()
                if text and len(text) > 10:  # Filter out matches that are too short
                    bhc_parts.append(text)
        
        # Extract DI-specific context
        for pattern, flags in di_patterns:
            matches = list(re.finditer(pattern, ehr_text, flags))
            for match in matches:
                text = match.group(0).strip()
                if text and len(text) > 5:
                    di_parts.append(text)
        
        # Deduplicate while preserving order
        seen_shared = set()
        seen_bhc = set()
        seen_di = set()
        
        shared_unique = []
        bhc_unique = []
        di_unique = []
        
        for part in shared_parts:
            if part not in seen_shared:
                seen_shared.add(part)
                shared_unique.append(part)
        
        for part in bhc_parts:
            if part not in seen_bhc:
                seen_bhc.add(part)
                bhc_unique.append(part)
        
        for part in di_parts:
            if part not in seen_di:
                seen_di.add(part)
                di_unique.append(part)
        
        # Combine results (preserving original format, separated by double newlines)
        shared_context = '\n\n'.join(shared_unique) if shared_unique else ""
        bhc_specific_context = '\n\n'.join(bhc_unique) if bhc_unique else ""
        di_specific_context = '\n\n'.join(di_unique) if di_unique else ""
        
        # If some sections are empty, use original text as fallback (to avoid losing information)
        if not shared_context and not bhc_specific_context and not di_specific_context:
            self.log("Warning: rule matching failed, falling back to raw text", "WARNING")
            # Simple split: first 1/3 as shared, middle 1/3 as BHC, last 1/3 as DI
            text_len = len(ehr_text)
            shared_context = ehr_text[:text_len//3]
            bhc_specific_context = ehr_text[text_len//3:2*text_len//3]
            di_specific_context = ehr_text[2*text_len//3:]
        
        return {
            "shared_context": shared_context,
            "bhc_specific_context": bhc_specific_context,
            "di_specific_context": di_specific_context,
            "raw_segmentation": "Rule-based segmentation using regex patterns"
        }
    
    def _fix_json_common_issues(self, text: str) -> str:
        """
        Fix common JSON issues
        """
        import re
        
        # Extract JSON portion
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
        else:
            json_str = text
        
        # Fix common JSON issues
        # 1. Fix single quotes (replace with double quotes)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # 2. Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 3. Fix unescaped special characters
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        # 4. Attempt to fix incomplete strings
        # If a string is not properly closed, attempt to fix
        lines = json_str.split('\n')
        fixed_lines = []
        for line in lines:
            # Check for unclosed strings
            if line.count('"') % 2 != 0:
                # Attempt to fix
                if not line.rstrip().endswith('"') and not line.rstrip().endswith('",'):
                    line = line.rstrip() + '"'
            fixed_lines.append(line)
        json_str = '\n'.join(fixed_lines)
        
        return json_str
    
    def _parse_fallback(self, text: str) -> Dict[str, Any]:
        """
        Fallback parser when JSON parsing fails
        Try to extract three sections from text
        """
        result = {
            "shared_context": "",
            "bhc_specific_context": "",
            "di_specific_context": ""
        }
        
        # Try to find keywords and extract content
        text_lower = text.lower()
        
        # Find shared context
        shared_patterns = [
            r'shared[_\s]*context["\']?\s*[:：]\s*["\']?([^"\']+)',
            r'"shared_context"\s*:\s*"([^"]+)"',
            r"'shared_context'\s*:\s*'([^']+)'",
        ]
        for pattern in shared_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["shared_context"] = match.group(1).strip()
                break
        
        # Find BHC-specific context
        bhc_patterns = [
            r'bhc[_\s]*specific[_\s]*context["\']?\s*[:：]\s*["\']?([^"\']+)',
            r'"bhc_specific_context"\s*:\s*"([^"]+)"',
            r"'bhc_specific_context'\s*:\s*'([^']+)'",
            r'brief[_\s]*hospital[_\s]*course[_\s]*specific[_\s]*context["\']?\s*[:：]\s*["\']?([^"\']+)',
        ]
        for pattern in bhc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["bhc_specific_context"] = match.group(1).strip()
                break
        
        # Find DI-specific context
        di_patterns = [
            r'di[_\s]*specific[_\s]*context["\']?\s*[:：]\s*["\']?([^"\']+)',
            r'"di_specific_context"\s*:\s*"([^"]+)"',
            r"'di_specific_context'\s*:\s*'([^']+)'",
            r'discharge[_\s]*instructions?[_\s]*specific[_\s]*context["\']?\s*[:：]\s*["\']?([^"\']+)',
        ]
        for pattern in di_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["di_specific_context"] = match.group(1).strip()
                break
        
        # If still not found, use simple splitting as a last resort
        if not any(result.values()):
            self.log("Using simple split as fallback", "WARNING")
            # Simple split: first 1/3 as shared, middle 1/3 as BHC, last 1/3 as DI
            text_len = len(text)
            result["shared_context"] = text[:text_len//3]
            result["bhc_specific_context"] = text[text_len//3:2*text_len//3]
            result["di_specific_context"] = text[2*text_len//3:]
        
        return result


