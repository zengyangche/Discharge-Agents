"""
Input Decoupling Agent: extracts BHC/DI-specific and shared inputs from raw files
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class InputDecouplingAgent(BaseAgent):
    """
    Decouple input file content into:
    - BHC-specific input (brief_hospital_course)
    - DI-specific input (discharge_instructions)
    - shared input (required by both)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("InputDecouplingAgent", config)
        
    def process(self, stay_id: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single stay_id and extract relevant inputs
        
        Args:
            stay_id: Hospital stay ID
            
        Returns:
            {
                "shared_input": {...},  # shared input
                "bhc_specific_input": {...},  # BHC-specific input
                "di_specific_input": {...},  # DI-specific input
                "raw_data": {...}  # raw data (for verification)
            }
        """
        try:
            # Load raw data
            raw_data = self._load_raw_data(stay_id)
            
            # Extract shared input
            shared_input = self._extract_shared_input(raw_data)
            
            # Extract BHC-specific input
            bhc_specific = self._extract_bhc_specific_input(raw_data)
            
            # Extract DI-specific input
            di_specific = self._extract_di_specific_input(raw_data)
            
            return {
                "shared_input": shared_input,
                "bhc_specific_input": bhc_specific,
                "di_specific_input": di_specific,
                "raw_data": raw_data,
                "stay_id": stay_id
            }
        except Exception as e:
            self.log(f"Error processing stay_id {stay_id}: {str(e)}", "ERROR")
            raise
    
    def _load_raw_data(self, stay_id: str) -> Dict[str, Any]:
        """Load all raw data for the given stay_id"""
        from utils.config import DIAGNOSIS_PATH, EDSTAYS_PATH, TRIAGE_PATH, DISCHARGE_PATH
        
        raw_data = {
            "stay_id": stay_id,
            "diagnosis": None,
            "edstays": None,
            "triage": None,
            "discharge": None,
        }
        
        # Load diagnosis
        try:
            df_diag = pd.read_csv(DIAGNOSIS_PATH)
            raw_data["diagnosis"] = df_diag[df_diag["stay_id"] == int(stay_id)].to_dict('records')
        except Exception as e:
            self.log(f"Failed to load diagnosis: {str(e)}", "WARNING")
        
        # Load edstays
        try:
            df_edstays = pd.read_csv(EDSTAYS_PATH)
            raw_data["edstays"] = df_edstays[df_edstays["stay_id"] == int(stay_id)].to_dict('records')
        except Exception as e:
            self.log(f"Failed to load edstays: {str(e)}", "WARNING")
        
        # Load triage
        try:
            df_triage = pd.read_csv(TRIAGE_PATH)
            raw_data["triage"] = df_triage[df_triage["stay_id"] == int(stay_id)].to_dict('records')
        except Exception as e:
            self.log(f"Failed to load triage: {str(e)}", "WARNING")
        
        # Load discharge note text
        try:
            df_discharge = pd.read_csv(DISCHARGE_PATH)
            discharge_records = df_discharge[df_discharge["note_id"].str.contains(stay_id, na=False)].to_dict('records')
            raw_data["discharge"] = discharge_records
        except Exception as e:
            self.log(f"Failed to load discharge: {str(e)}", "WARNING")
        
        return raw_data
    
    def _extract_shared_input(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract shared input (needed by both BHC and DI)
        - Patient demographics (from edstays)
        - Primary diagnoses (from diagnosis)
        - Admission info (from edstays)
        """
        shared = {
            "patient_info": {},
            "primary_diagnosis": [],
            "admission_info": {},
        }
        
        # Extract patient info and admission info from edstays
        if raw_data.get("edstays"):
            edstay = raw_data["edstays"][0] if raw_data["edstays"] else {}
            shared["patient_info"] = {
                "subject_id": edstay.get("subject_id"),
                "gender": edstay.get("gender"),
                "race": edstay.get("race"),
                "arrival_transport": edstay.get("arrival_transport"),
            }
            shared["admission_info"] = {
                "intime": edstay.get("intime"),
                "outtime": edstay.get("outtime"),
                "disposition": edstay.get("disposition"),
            }
        
        # Extract primary diagnoses from diagnosis
        if raw_data.get("diagnosis"):
            shared["primary_diagnosis"] = [
                {
                    "icd_code": d.get("icd_code"),
                    "icd_title": d.get("icd_title"),
                    "seq_num": d.get("seq_num"),
                }
                for d in raw_data["diagnosis"][:5]  # Top 5 diagnoses
            ]
        
        return shared
    
    def _extract_bhc_specific_input(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract BHC-specific input
        - Full diagnosis list (from diagnosis)
        - Radiology reports (from radiology, if any)
        - Triage info (from triage)
        - Full discharge note text (from discharge)
        """
        bhc_specific = {
            "all_diagnoses": [],
            "triage_info": {},
            "discharge_note_text": "",
        }
        
        # All diagnoses
        if raw_data.get("diagnosis"):
            bhc_specific["all_diagnoses"] = [
                {
                    "icd_code": d.get("icd_code"),
                    "icd_title": d.get("icd_title"),
                    "seq_num": d.get("seq_num"),
                    "icd_version": d.get("icd_version"),
                }
                for d in raw_data["diagnosis"]
            ]
        
        # Triage information
        if raw_data.get("triage"):
            triage = raw_data["triage"][0] if raw_data["triage"] else {}
            bhc_specific["triage_info"] = {
                "chiefcomplaint": triage.get("chiefcomplaint"),
                "temperature": triage.get("temperature"),
                "heartrate": triage.get("heartrate"),
                "resprate": triage.get("resprate"),
                "o2sat": triage.get("o2sat"),
                "sbp": triage.get("sbp"),
                "dbp": triage.get("dbp"),
                "pain": triage.get("pain"),
                "acuity": triage.get("acuity"),
            }
        
        # Discharge note text (used as reference for BHC generation)
        if raw_data.get("discharge"):
            discharge_texts = [d.get("text", "") for d in raw_data["discharge"] if d.get("text")]
            bhc_specific["discharge_note_text"] = "\n\n".join(discharge_texts)
        
        return bhc_specific
    
    def _extract_di_specific_input(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract DI-specific input
        - Diagnosis-related medication needs
        - Discharge disposition (from edstays)
        - Patient education needs (inferred from diagnosis and triage)
        """
        di_specific = {
            "disposition": "",
            "medication_needs": [],
            "followup_needs": [],
        }
        
        # Discharge disposition
        if raw_data.get("edstays"):
            edstay = raw_data["edstays"][0] if raw_data["edstays"] else {}
            di_specific["disposition"] = edstay.get("disposition", "")
        
        # Infer medication needs based on diagnoses (simple logic, can be optimized later)
        if raw_data.get("diagnosis"):
            diagnoses = [d.get("icd_title", "").lower() for d in raw_data["diagnosis"][:3]]
            # More complex logic can be added here to infer medication and follow-up needs
            # For example: if diagnosis contains "fracture", "orthopedic follow-up" may be needed
        
        return di_specific

