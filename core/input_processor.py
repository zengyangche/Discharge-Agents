"""
Input Processor (Decoupling Agent)
Decouples raw input into shared context, BHC context, and DI context.
"""
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InputProcessor:
    """
    Input processing and decoupling agent.

    Features:
    1. Load raw data from multiple sources
    2. Decouple data into:
       - Shared_Context: shared context (diagnoses, patient demographics, etc.)
       - BHC_Context: BHC-specific context (course, exam results, etc.)
       - DI_Context: DI-specific context (medications, follow-up, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load data using DataLoader
        try:
            from utils.data_loader import DataLoader
            self.data_loader = DataLoader()
        except ImportError:
            self.logger.warning("DataLoader unavailable, falling back to legacy data loading")
            self.data_loader = None
    
    def process(self, stay_id: str) -> Dict[str, Any]:
        """
        Process input data for the given stay_id.

        Args:
            stay_id: Hospital stay ID
            
        Returns:
            {
                "shared_context": {...},
                "bhc_context": {...},
                "di_context": {...},
                "raw_data": {...}  # keep raw data for verification
            }
        """
        try:
            # Load raw data
            raw_data = self._load_raw_data(stay_id)
            
            # Decouple into different contexts
            shared_context = self._extract_shared_context(raw_data)
            bhc_context = self._extract_bhc_context(raw_data)
            di_context = self._extract_di_context(raw_data)
            
            return {
                "shared_context": shared_context,
                "bhc_context": bhc_context,
                "di_context": di_context,
                "raw_data": raw_data,
                "stay_id": stay_id
            }
        except Exception as e:
            self.logger.error(f"Error processing stay_id {stay_id}: {str(e)}")
            raise
    
    def _load_raw_data(self, stay_id: Union[str, int]) -> Dict[str, Any]:
        """Load all raw data for the given stay_id"""
        # Prefer DataLoader
        if self.data_loader is not None:
            try:
                loaded_data = self.data_loader.load_stay_data(stay_id)
                if loaded_data["has_data"]:
                    return loaded_data
                else:
                    self.logger.warning(f"No data found for stay_id {stay_id}")
            except Exception as e:
                self.logger.warning(f"DataLoader failed, falling back to legacy method: {str(e)}")
        
        # Fall back to legacy method (compatibility)
        from utils.config import DIAGNOSIS_PATH, EDSTAYS_PATH, TRIAGE_PATH, DISCHARGE_PATH
        
        raw_data = {
            "stay_id": str(stay_id),
            "diagnosis": [],
            "edstays": {},
            "triage": {},
            "discharge": [],
        }
        
        # Load diagnosis
        try:
            if DIAGNOSIS_PATH.exists():
                df_diag = pd.read_csv(DIAGNOSIS_PATH, low_memory=False)
                stay_id_int = int(stay_id) if str(stay_id).isdigit() else None
                if stay_id_int is not None:
                    diagnosis_records = df_diag[df_diag["stay_id"] == stay_id_int].to_dict('records')
                else:
                    diagnosis_records = df_diag[df_diag["stay_id"].astype(str) == str(stay_id)].to_dict('records')
                raw_data["diagnosis"] = diagnosis_records
        except Exception as e:
            self.logger.warning(f"Failed to load diagnosis: {str(e)}")
        
        # Load edstays
        try:
            if EDSTAYS_PATH.exists():
                df_edstays = pd.read_csv(EDSTAYS_PATH, low_memory=False)
                stay_id_int = int(stay_id) if str(stay_id).isdigit() else None
                if stay_id_int is not None:
                    edstay_record = df_edstays[df_edstays["stay_id"] == stay_id_int]
                else:
                    edstay_record = df_edstays[df_edstays["stay_id"].astype(str) == str(stay_id)]
                if not edstay_record.empty:
                    raw_data["edstays"] = edstay_record.iloc[0].to_dict()
        except Exception as e:
            self.logger.warning(f"Failed to load edstays: {str(e)}")
        
        # Load triage
        try:
            if TRIAGE_PATH.exists():
                df_triage = pd.read_csv(TRIAGE_PATH, low_memory=False)
                stay_id_int = int(stay_id) if str(stay_id).isdigit() else None
                if stay_id_int is not None:
                    triage_record = df_triage[df_triage["stay_id"] == stay_id_int]
                else:
                    triage_record = df_triage[df_triage["stay_id"].astype(str) == str(stay_id)]
                if not triage_record.empty:
                    raw_data["triage"] = triage_record.iloc[0].to_dict()
        except Exception as e:
            self.logger.warning(f"Failed to load triage: {str(e)}")
        
        # Load discharge (discharge summary text)
        try:
            if DISCHARGE_PATH.exists():
                df_discharge = pd.read_csv(DISCHARGE_PATH, low_memory=False)
                hadm_id = raw_data["edstays"].get("hadm_id")
                if hadm_id:
                    hadm_id_int = int(hadm_id) if str(hadm_id).isdigit() else None
                    # Try different ID column names
                    for id_col in ["hadm_id", "note_id", "stay_id"]:
                        if id_col in df_discharge.columns:
                            if hadm_id_int is not None:
                                discharge_records = df_discharge[df_discharge[id_col] == hadm_id_int].to_dict('records')
                            else:
                                discharge_records = df_discharge[df_discharge[id_col].astype(str) == str(hadm_id)].to_dict('records')
                            raw_data["discharge"] = discharge_records
                            break
        except Exception as e:
            self.logger.warning(f"Failed to load discharge: {str(e)}")
        
        return raw_data
    
    def _extract_shared_context(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract shared context (information needed by both BHC and DI).

        Includes:
        - Patient demographics
        - Primary diagnoses
        - Admission/discharge information
        """
        shared = {
            "patient_info": {},
            "primary_diagnoses": [],
            "admission_info": {},
        }
        
        # Patient information
        edstays = raw_data.get("edstays", {})
        if edstays:
            shared["patient_info"] = {
                "subject_id": edstays.get("subject_id"),
                "gender": edstays.get("gender"),
                "race": edstays.get("race"),
                "arrival_transport": edstays.get("arrival_transport"),
            }
            shared["admission_info"] = {
                "intime": edstays.get("intime"),
                "outtime": edstays.get("outtime"),
                "disposition": edstays.get("disposition"),
            }
        
        # Primary diagnoses (top 5)
        diagnoses = raw_data.get("diagnosis", [])
        shared["primary_diagnoses"] = [
            {
                "icd_code": d.get("icd_code"),
                "icd_title": d.get("icd_title"),
                "seq_num": d.get("seq_num"),
                "icd_version": d.get("icd_version"),
            }
            for d in sorted(diagnoses, key=lambda x: x.get("seq_num", 999))[:5]
        ]
        
        return shared
    
    def _extract_bhc_context(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract BHC-specific context.

        Includes:
        - Full diagnosis list
        - Triage information (vitals, chief complaint, etc.)
        - Raw discharge note text (for reference)
        - Possible clinical findings
        """
        bhc_context = {
            "all_diagnoses": [],
            "triage_info": {},
            "discharge_note_text": "",
            "clinical_notes": [],
        }
        
        # All diagnoses
        diagnoses = raw_data.get("diagnosis", [])
        bhc_context["all_diagnoses"] = [
            {
                "icd_code": d.get("icd_code"),
                "icd_title": d.get("icd_title"),
                "seq_num": d.get("seq_num"),
                "icd_version": d.get("icd_version"),
            }
            for d in sorted(diagnoses, key=lambda x: x.get("seq_num", 999))
        ]
        
        # Triage information
        triage = raw_data.get("triage", {})
        if triage:
            bhc_context["triage_info"] = {
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
        
        # Discharge note text (for reference)
        discharge_records = raw_data.get("discharge", [])
        if discharge_records:
            texts = [d.get("text", "") for d in discharge_records if d.get("text")]
            bhc_context["discharge_note_text"] = "\n\n".join(texts)
        
        return bhc_context
    
    def _extract_di_context(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract DI-specific context.

        Includes:
        - Discharge disposition
        - Inferred medication needs from diagnoses
        - Possible follow-up needs
        """
        di_context = {
            "disposition": "",
            "medication_needs": [],
            "followup_needs": [],
            "activity_restrictions": [],
        }
        
        # Discharge disposition
        edstays = raw_data.get("edstays", {})
        if edstays:
            di_context["disposition"] = edstays.get("disposition", "")
        
        # Infer needs based on diagnoses (simplified logic; should be more complex in practice)
        diagnoses = raw_data.get("diagnosis", [])
        diagnosis_titles = [d.get("icd_title", "").lower() for d in diagnoses[:3]]
        
        # Simple rule-based inference
        if any("fracture" in d for d in diagnosis_titles):
            di_context["activity_restrictions"].append("Weight bearing restrictions")
            di_context["followup_needs"].append("Orthopedic follow-up")
        
        if any("diabetes" in d or "diabetic" in d for d in diagnosis_titles):
            di_context["medication_needs"].append("Blood glucose monitoring")
        
        if any("anticoagulation" in d or "thrombosis" in d for d in diagnosis_titles):
            di_context["medication_needs"].append("Anticoagulation medication")
        
        return di_context
