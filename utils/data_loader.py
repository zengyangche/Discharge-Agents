"""
Data loading utility
Reads raw data from CSV files; supports single stay_id or batch loading
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd

# Ensure project-local config.py is used instead of the system config package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    DIAGNOSIS_PATH,
    EDSTAYS_PATH,
    TRIAGE_PATH,
    DISCHARGE_PATH,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader: reads data from CSV files.

    Features:
    1. Load data for a single stay_id
    2. Batch-load multiple stay_ids
    3. List all available stay_ids
    4. Handle missing data files gracefully
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        diagnosis_path: Optional[Path] = None,
        edstays_path: Optional[Path] = None,
        triage_path: Optional[Path] = None,
        discharge_path: Optional[Path] = None
    ):
        """
        Args:
            data_dir: Data directory path; defaults to DATA_DIR in config
            diagnosis_path: Custom diagnosis file path (optional)
            edstays_path: Custom ED stay file path (optional)
            triage_path: Custom triage file path (optional)
            discharge_path: Custom discharge note file path (optional)
        """
        self.data_dir = data_dir or DATA_DIR
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Data file paths (support custom paths)
        # If no custom path is provided, use the paths configured in config.py
        self.diagnosis_path = diagnosis_path or DIAGNOSIS_PATH
        self.edstays_path = edstays_path or EDSTAYS_PATH
        self.triage_path = triage_path or TRIAGE_PATH
        self.discharge_path = discharge_path or DISCHARGE_PATH
        
        # Data cache (optional, improves performance in batch processing)
        self._cache = {}
        self._cache_enabled = False
    
    def enable_cache(self, enable: bool = True):
        """Enable/disable data cache"""
        self._cache_enabled = enable
        if not enable:
            self._cache.clear()
    
    def _get_stay_id_from_hadm_id(self, hadm_id: Union[str, int]) -> Optional[str]:
        """
        Look up stay_id from hadm_id.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            Corresponding stay_id, or None if not found
        """
        if not self.edstays_path.exists():
            return None
        
        try:
            cache_key = f"hadm_to_stay_{hadm_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            df = pd.read_csv(
                self.edstays_path,
                usecols=["stay_id", "hadm_id"],
                dtype={"stay_id": "Int64", "hadm_id": "Int64"},
            )
            
            hadm_id_int = int(hadm_id) if str(hadm_id).isdigit() else None
            if hadm_id_int is not None:
                df_filtered = df[df["hadm_id"] == hadm_id_int]
            else:
                df_filtered = df[df["hadm_id"].astype(str) == str(hadm_id)]
            
            if df_filtered.empty:
                return None
            
            # A hadm_id may correspond to multiple stay_ids; take the first one
            stay_id = str(df_filtered.iloc[0]["stay_id"])
            
            if self._cache_enabled:
                self._cache[cache_key] = stay_id
            
            return stay_id
            
        except Exception as e:
            self.logger.error(f"Failed to look up stay_id from hadm_id: {str(e)}")
            return None
    
    def _get_hadm_id_from_stay_id(self, stay_id: Union[str, int]) -> Optional[str]:
        """
        Look up hadm_id from stay_id.

        Args:
            stay_id: ED stay ID

        Returns:
            Corresponding hadm_id, or None if not found
        """
        edstays_data = self._load_edstays(str(stay_id))
        return str(edstays_data.get("hadm_id")) if edstays_data.get("hadm_id") else None
    
    def load_stay_data(
        self, 
        stay_id: Optional[Union[str, int]] = None,
        hadm_id: Optional[Union[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Load all related data for the given stay_id or hadm_id.

        Args:
            stay_id: ED stay ID (optional if hadm_id is provided)
            hadm_id: Hospital admission ID (optional if stay_id is provided)

        Returns:
            {
                "stay_id": str,
                "hadm_id": str or None,
                "diagnosis": List[Dict],
                "edstays": Dict,
                "triage": Dict,
                "discharge": List[Dict],
                "has_data": bool,  # whether any data exists
            }

        Raises:
            ValueError: If neither stay_id nor hadm_id is provided
        """
        # Determine stay_id
        if hadm_id is not None:
            # If hadm_id is provided, first look up the corresponding stay_id
            stay_id = self._get_stay_id_from_hadm_id(hadm_id)
            if stay_id is None:
                self.logger.warning(f"No stay_id found for hadm_id {hadm_id}")
                stay_id = None
        elif stay_id is None:
            raise ValueError("Either stay_id or hadm_id must be provided")
        
        stay_id = str(stay_id) if stay_id else None
        
        data = {
            "stay_id": stay_id,
            "hadm_id": None,
            "diagnosis": [],
            "edstays": {},
            "triage": {},
            "discharge": [],
            "has_data": False,
        }
        
        try:
            if stay_id:
                # Load diagnosis
                data["diagnosis"] = self._load_diagnosis(stay_id)
                
                # Load edstays
                data["edstays"] = self._load_edstays(stay_id)
                
                # Load triage
                data["triage"] = self._load_triage(stay_id)
                
                # Get hadm_id from edstays
                if data["edstays"].get("hadm_id"):
                    data["hadm_id"] = str(data["edstays"]["hadm_id"])
            
            # Load discharge (using hadm_id)
            hadm_id_to_use = hadm_id if hadm_id is not None else data["hadm_id"]
            if hadm_id_to_use:
                data["discharge"] = self._load_discharge(hadm_id_to_use)
                # If hadm_id was not set before, set it now
                if not data["hadm_id"]:
                    data["hadm_id"] = str(hadm_id_to_use)
            
            # Check if any data exists
            data["has_data"] = (
                len(data["diagnosis"]) > 0 or
                len(data["edstays"]) > 0 or
                len(data["triage"]) > 0 or
                len(data["discharge"]) > 0
            )
            
        except Exception as e:
            self.logger.error(f"Error loading data (stay_id={stay_id}, hadm_id={hadm_id}): {str(e)}")
            raise
        
        return data
    
    def _load_diagnosis(self, stay_id: str) -> List[Dict[str, Any]]:
        """Load diagnosis data"""
        if not self.diagnosis_path.exists():
            self.logger.warning(f"Diagnosis file not found: {self.diagnosis_path}")
            return []
        
        try:
            # Use cache
            cache_key = f"diagnosis_{stay_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            # Read CSV (use chunksize to handle large files)
            df = pd.read_csv(
                self.diagnosis_path,
                dtype={"stay_id": "Int64"},  # Use nullable integer type
            )
            
            # Filter by specified stay_id
            stay_id_int = int(stay_id) if stay_id.isdigit() else None
            if stay_id_int is not None:
                df_filtered = df[df["stay_id"] == stay_id_int]
            else:
                df_filtered = df[df["stay_id"].astype(str) == stay_id]
            
            result = df_filtered.to_dict('records')
            
            # Cache result
            if self._cache_enabled:
                self._cache[cache_key] = result
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to read diagnosis file: {str(e)}")
            return []

    def _load_edstays(self, stay_id: str) -> Dict[str, Any]:
        """Load ED stay data"""
        if not self.edstays_path.exists():
            self.logger.warning(f"ED stays file not found: {self.edstays_path}")
            return {}

        try:
            cache_key = f"edstays_{stay_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]

            df = pd.read_csv(
                self.edstays_path,
                dtype={"stay_id": "Int64"}
            )

            stay_id_int = int(stay_id) if stay_id.isdigit() else None
            if stay_id_int is not None:
                df_filtered = df[df["stay_id"] == stay_id_int]
            else:
                df_filtered = df[df["stay_id"].astype(str) == stay_id]

            if df_filtered.empty:
                return {}

            result = df_filtered.iloc[0].to_dict()
            result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

            if self._cache_enabled:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Failed to read ED stays file: {str(e)}")
            return {}

    def _load_triage(self, stay_id: str) -> Dict[str, Any]:
        """Load triage data"""
        if not self.triage_path.exists():
            self.logger.warning(f"Triage file not found: {self.triage_path}")
            return {}

        try:
            cache_key = f"triage_{stay_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]

            df = pd.read_csv(
                self.triage_path,
                dtype={"stay_id": "Int64"}
            )

            stay_id_int = int(stay_id) if stay_id.isdigit() else None
            if stay_id_int is not None:
                df_filtered = df[df["stay_id"] == stay_id_int]
            else:
                df_filtered = df[df["stay_id"].astype(str) == stay_id]

            if df_filtered.empty:
                return {}

            result = df_filtered.iloc[0].to_dict()
            result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

            if self._cache_enabled:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Failed to read triage file: {str(e)}")
            return {}

    def _load_discharge(self, hadm_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Load discharge note text (fault-tolerant read)"""
        if not self.discharge_path.exists():
            self.logger.warning(f"Discharge note file not found: {self.discharge_path}")
            return []

        try:
            hadm_id_str = str(hadm_id)
            cache_key = f"discharge_{hadm_id_str}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]

            # First read column names to determine ID column
            df_sample = pd.read_csv(self.discharge_path, nrows=1, engine="python")
            id_column = None
            for col in ["hadm_id", "note_id", "stay_id"]:
                if col in df_sample.columns:
                    id_column = col
                    break
            if id_column is None:
                self.logger.warning("No ID column found in discharge note file")
                return []

            # Fault-tolerant read: read all as strings, skip bad lines, avoid parse errors
            df = pd.read_csv(
                self.discharge_path,
                dtype=str,
                on_bad_lines="skip",
                engine="python",
            )

            df_filtered = df[df[id_column].astype(str) == hadm_id_str]

            result = df_filtered.to_dict("records")
            # Handle missing values
            for record in result:
                for k, v in record.items():
                    if pd.isna(v):
                        record[k] = None

            if self._cache_enabled:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Failed to read discharge note file: {str(e)}")
            return []

    # ---------------------------
    #  Static utility: read IDs from CSV
    # ---------------------------
    @staticmethod
    def load_stay_ids_from_csv(
        csv_path: Union[str, Path],
        stay_id_column: Optional[str] = None,
        hadm_id_column: Optional[str] = None,
        data_dir: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Read stay_id list from the given CSV file.
        Supports stay_id or hadm_id columns (hadm_id is auto-converted to stay_id).

        Args:
            csv_path: CSV file path
            stay_id_column: stay_id column name (optional, if CSV has stay_id)
            hadm_id_column: hadm_id column name (optional; auto-converts to stay_id)
            data_dir: Data directory (for edstays.csv ID conversion)
            limit: Maximum number of stay_ids to return

        Returns:
            List of stay_ids
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"CSV file not found: {csv_path}")
            return []

        try:
            _logger = logging.getLogger(__name__)

            # First read column names to check if columns exist
            df_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

            # Determine which column to read
            id_column = None
            use_hadm_id = False

            if stay_id_column and stay_id_column in df_columns:
                id_column = stay_id_column
                use_hadm_id = False
            elif hadm_id_column and hadm_id_column in df_columns:
                id_column = hadm_id_column
                use_hadm_id = True
            else:
                # Auto-detect: prefer stay_id, fall back to hadm_id
                if "stay_id" in df_columns:
                    id_column = "stay_id"
                    use_hadm_id = False
                    _logger.info(f"Auto-detected stay_id column: {id_column}")
                elif "hadm_id" in df_columns:
                    id_column = "hadm_id"
                    use_hadm_id = True
                    _logger.info(f"Auto-detected hadm_id column: {id_column}, will convert to stay_id")
                else:
                    _logger.warning(
                        f"No stay_id or hadm_id column found in {csv_path}. "
                        f"Available columns: {', '.join(df_columns)}"
                    )
                    return []

            # Read CSV file (only read needed columns for performance)
            df = pd.read_csv(
                csv_path,
                usecols=[id_column],
                dtype={id_column: "Int64"}
            )

            # Extract IDs
            ids = df[id_column].dropna().unique().astype(str).tolist()

            # If conversion from hadm_id to stay_id is needed
            if use_hadm_id:
                _logger.info(f"Converting hadm_id to stay_id...")
                loader = DataLoader(data_dir=data_dir)
                stay_ids = []
                for hadm_id in ids:
                    stay_id = loader._get_stay_id_from_hadm_id(hadm_id)
                    if stay_id:
                        stay_ids.append(stay_id)
                    else:
                        _logger.warning(f"No stay_id found for hadm_id {hadm_id}")
                ids = stay_ids
                _logger.info(f"Successfully converted {len(ids)} hadm_ids to stay_ids")

            if limit:
                ids = ids[:limit]

            _logger.info(f"Read {len(ids)} stay_ids from {csv_path}")
            return sorted(ids)

        except Exception as e:
            _logger = logging.getLogger(__name__)
            _logger.error(f"Failed to read stay_ids from CSV: {str(e)}")
            return []

    def _load_radiology(
        self,
        stay_id: Optional[str] = None,
        hadm_id: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Load radiology reports"""
        path = self.data_dir / "radiology.csv"
        if not path.exists():
            return []

        try:
            cache_key = f"radiology_{stay_id}_{hadm_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]

            df = pd.read_csv(path)

            df_filtered = df
            if hadm_id is not None:
                hadm_id_int = int(hadm_id) if str(hadm_id).isdigit() else None
                if "hadm_id" in df.columns:
                    if hadm_id_int is not None:
                        df_filtered = df_filtered[df_filtered["hadm_id"] == hadm_id_int]
                    else:
                        df_filtered = df_filtered[df_filtered["hadm_id"].astype(str) == str(hadm_id)]
            if stay_id is not None and "stay_id" in df.columns:
                stay_id_int = int(stay_id) if str(stay_id).isdigit() else None
                if stay_id_int is not None:
                    df_filtered = df_filtered[df_filtered["stay_id"] == stay_id_int]
                else:
                    df_filtered = df_filtered[df_filtered["stay_id"].astype(str) == str(stay_id)]

            result = df_filtered.to_dict("records")
            result = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in result]

            if self._cache_enabled:
                self._cache[cache_key] = result

            return result
        except Exception as e:
            self.logger.error(f"Failed to read radiology file: {str(e)}")
            return []

    def load_case_inputs(
        self,
        stay_id: Optional[Union[str, int]] = None,
        hadm_id: Optional[Union[str, int]] = None,
        drop_discharge_labels: bool = True,
        strip_bhc_di_sections: bool = True,
    ) -> Dict[str, Any]:
        """
        Load raw inputs for a single case (zero-shot, no labels).

        Data sources:
        - diagnosis.csv   (by stay_id)
        - edstays.csv     (stay_id, hadm_id, subject_id)
        - triage.csv      (by stay_id)
        - radiology.csv   (hadm_id preferred, else stay_id)
        - discharge.csv   (by hadm_id; strip possible BHC/DI label columns)

        Args:
            stay_id: ED stay ID
            hadm_id: Hospital admission ID
            drop_discharge_labels: Remove possible BHC/DI label columns from discharge.csv

        Returns:
            {
                "ids": {"stay_id": ..., "hadm_id": ..., "subject_id": ...},
                "edstays": {...},
                "diagnosis": [...],
                "triage": {...},
                "radiology": [...],
                "discharge": [...],
            }
        """
        # Load base data using the parent method first
        base = self.load_stay_data(stay_id=stay_id, hadm_id=hadm_id)

        # radiology prefers hadm_id
        rad = self._load_radiology(
            stay_id=base.get("stay_id"),
            hadm_id=base.get("hadm_id"),
        )

        discharge_records = base.get("discharge", [])
        if drop_discharge_labels and discharge_records:
            # Remove possible BHC/DI label columns
            remove_cols = {
                "discharge_instructions",
                "brief_hospital_course",
                "discharge_instructions_word_count",
                "brief_hospital_course_word_count",
            }
            cleaned = []
            for r in discharge_records:
                cleaned.append({k: v for k, v in r.items() if k not in remove_cols})
            discharge_records = cleaned

        if strip_bhc_di_sections and discharge_records:
            # Strip BHC/DI sections from discharge note text
            for r in discharge_records:
                if "text" in r and isinstance(r["text"], str):
                    r["text"] = self._strip_bhc_di_sections(r["text"])

        return {
            "ids": {
                "stay_id": base.get("stay_id"),
                "hadm_id": base.get("hadm_id"),
                "subject_id": base.get("edstays", {}).get("subject_id"),
            },
            "edstays": base.get("edstays"),
            "diagnosis": base.get("diagnosis", []),
            "triage": base.get("triage"),
            "radiology": rad,
            "discharge": discharge_records,
        }

    @staticmethod
    def _strip_bhc_di_sections(text: str) -> str:
        """
        Remove BHC / DI sections from discharge note text to avoid label leakage.

        Section headings removed (case-insensitive, surrounding whitespace ignored):
        - Brief Hospital Course
        - Discharge Instructions

        Simple strategy: from each matched heading, delete until the next
        all-caps/title line or end of text.
        """
        import re

        # Define section headings to remove
        sections = [
            r"Brief Hospital Course",
            r"Discharge Instructions",
        ]

        # Build regex to match from heading to next heading or end of text
        pattern = r"(?ims)" + "|".join(
            rf"{sec}\s*:?.*?(?=\n[A-Z][A-Za-z ]{{2,}}:|\Z)" for sec in sections
        )

        stripped = re.sub(pattern, "", text)
        return stripped.strip()


    
    def _load_edstays(self, stay_id: str) -> Dict[str, Any]:
        """Load ED stay data"""
        if not self.edstays_path.exists():
            self.logger.warning(f"ED stays file not found: {self.edstays_path}")
            return {}
        
        try:
            cache_key = f"edstays_{stay_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            df = pd.read_csv(
                self.edstays_path,
                dtype={"stay_id": "Int64"}
            )
            
            stay_id_int = int(stay_id) if stay_id.isdigit() else None
            if stay_id_int is not None:
                df_filtered = df[df["stay_id"] == stay_id_int]
            else:
                df_filtered = df[df["stay_id"].astype(str) == stay_id]
            
            if df_filtered.empty:
                return {}
            
            result = df_filtered.iloc[0].to_dict()
            
            # Handle NaN values, convert to None
            result = {k: (None if pd.isna(v) else v) for k, v in result.items()}
            
            if self._cache_enabled:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read ED stays file: {str(e)}")
            return {}
    
    def _load_triage(self, stay_id: str) -> Dict[str, Any]:
        """Load triage data"""
        if not self.triage_path.exists():
            self.logger.warning(f"Triage file not found: {self.triage_path}")
            return {}
        
        try:
            cache_key = f"triage_{stay_id}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            df = pd.read_csv(
                self.triage_path,
                dtype={"stay_id": "Int64"}
            )
            
            stay_id_int = int(stay_id) if stay_id.isdigit() else None
            if stay_id_int is not None:
                df_filtered = df[df["stay_id"] == stay_id_int]
            else:
                df_filtered = df[df["stay_id"].astype(str) == stay_id]
            
            if df_filtered.empty:
                return {}
            
            result = df_filtered.iloc[0].to_dict()
            result = {k: (None if pd.isna(v) else v) for k, v in result.items()}
            
            if self._cache_enabled:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read triage file: {str(e)}")
            return {}
    
    def _load_discharge(self, hadm_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Load discharge note text (fault-tolerant)"""
        if not self.discharge_path.exists():
            self.logger.warning(f"Discharge note file not found: {self.discharge_path}")
            return []
        
        try:
            hadm_id_str = str(hadm_id)
            cache_key = f"discharge_{hadm_id_str}"
            if self._cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            # First read column names to determine ID column
            df_sample = pd.read_csv(self.discharge_path, nrows=1, engine="python")
            id_column = None
            for col in ["hadm_id", "note_id", "stay_id"]:
                if col in df_sample.columns:
                    id_column = col
                    break
            
            if id_column is None:
                self.logger.warning("No ID column found in discharge note file")
                return []
            
            # Fault-tolerant read: read all as strings, skip bad lines (python engine does not support low_memory)
            df = pd.read_csv(
                self.discharge_path,
                dtype=str,
                on_bad_lines="skip",
                engine="python",
            )
            
            df_filtered = df[df[id_column].astype(str) == hadm_id_str]
            
            result = df_filtered.to_dict("records")
            for record in result:
                for k, v in record.items():
                    if pd.isna(v):
                        record[k] = None
            
            if self._cache_enabled:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read discharge note file: {str(e)}")
            return []
    
    def list_available_stay_ids(
        self,
        limit: Optional[int] = None,
        min_diagnosis_count: int = 0
    ) -> List[str]:
        """
        List all available stay_ids.

        Args:
            limit: Maximum number of stay_ids to return
            min_diagnosis_count: Minimum diagnosis count (for filtering)

        Returns:
            List of stay_ids
        """
        if not self.edstays_path.exists():
            self.logger.warning("ED stays file not found, cannot list stay_ids")
            return []
        
        try:
            df = pd.read_csv(
                self.edstays_path,
                usecols=["stay_id"],
                dtype={"stay_id": "Int64"}
            )
            
            stay_ids = df["stay_id"].dropna().unique().astype(str).tolist()
            
            # If minimum diagnosis count is set, filter
            if min_diagnosis_count > 0 and self.diagnosis_path.exists():
                df_diag = pd.read_csv(
                    self.diagnosis_path,
                    usecols=["stay_id"],
                    dtype={"stay_id": "Int64"}
                )
                diag_counts = df_diag["stay_id"].value_counts()
                stay_ids = [
                    sid for sid in stay_ids
                    if diag_counts.get(int(sid), 0) >= min_diagnosis_count
                ]
            
            if limit:
                stay_ids = stay_ids[:limit]
            
            return sorted(stay_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to list stay_ids: {str(e)}")
            return []
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get data statistics.

        Returns:
            Dict with file existence, record counts, and other statistics
        """
        stats = {
            "files_exist": {},
            "record_counts": {},
            "available_stay_ids": 0,
        }
        
        # Check file existence
        stats["files_exist"] = {
            "diagnosis": self.diagnosis_path.exists(),
            "edstays": self.edstays_path.exists(),
            "triage": self.triage_path.exists(),
            "discharge": self.discharge_path.exists(),
        }
        
        # Count records
        for name, path in [
            ("diagnosis", self.diagnosis_path),
            ("edstays", self.edstays_path),
            ("triage", self.triage_path),
            ("discharge", self.discharge_path),
        ]:
            if path.exists():
                try:
                    # Only read row count, do not load all data
                    df = pd.read_csv(path, usecols=[0])
                    stats["record_counts"][name] = len(df)
                except Exception as e:
                    self.logger.warning(f"Unable to count records in {name} file: {str(e)}")
                    stats["record_counts"][name] = None
            else:
                stats["record_counts"][name] = 0
        
        # Count available stay_ids
        try:
            stats["available_stay_ids"] = len(self.list_available_stay_ids())
        except Exception as e:
            self.logger.warning(f"Unable to count stay_ids: {str(e)}")
        
        return stats
    
    def validate_stay_id(self, stay_id: Union[str, int]) -> bool:
        """
        Check whether stay_id exists.

        Args:
            stay_id: Hospital stay ID

        Returns:
            True if data exists for this stay_id
        """
        data = self.load_stay_data(stay_id=stay_id)
        return data["has_data"]
    
    @staticmethod
    def load_stay_ids_from_csv(
        csv_path: Union[str, Path],
        stay_id_column: Optional[str] = None,
        hadm_id_column: Optional[str] = None,
        data_dir: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Read stay_id list from the given CSV file.
        Supports stay_id or hadm_id columns (hadm_id is auto-converted to stay_id).

        Args:
            csv_path: CSV file path
            stay_id_column: stay_id column name (optional, if CSV has stay_id)
            hadm_id_column: hadm_id column name (optional; auto-converts to stay_id)
            data_dir: Data directory (for edstays.csv ID conversion)
            limit: Maximum number of stay_ids to return

        Returns:
            List of stay_ids

        Example:
            >>> # Read from stay_id column
            >>> stay_ids = DataLoader.load_stay_ids_from_csv("data/test.csv", stay_id_column="stay_id")
            >>> # Read from hadm_id column (auto-converts to stay_id)
            >>> stay_ids = DataLoader.load_stay_ids_from_csv("data/discharge_target_test.csv", hadm_id_column="hadm_id")
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"CSV file not found: {csv_path}")
            return []
        
        try:
            _logger = logging.getLogger(__name__)
            
            # First read column names to check if columns exist
            df_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
            
            # Determine which column to read
            id_column = None
            use_hadm_id = False
            
            if stay_id_column and stay_id_column in df_columns:
                id_column = stay_id_column
                use_hadm_id = False
            elif hadm_id_column and hadm_id_column in df_columns:
                id_column = hadm_id_column
                use_hadm_id = True
            else:
                # Auto-detect: prefer stay_id, fall back to hadm_id
                if "stay_id" in df_columns:
                    id_column = "stay_id"
                    use_hadm_id = False
                    _logger.info(f"Auto-detected stay_id column: {id_column}")
                elif "hadm_id" in df_columns:
                    id_column = "hadm_id"
                    use_hadm_id = True
                    _logger.info(f"Auto-detected hadm_id column: {id_column}, will convert to stay_id")
                else:
                    _logger.warning(
                        f"No stay_id or hadm_id column found in {csv_path}. "
                        f"Available columns: {', '.join(df_columns)}"
                    )
                    return []
            
            # Read CSV file (only read needed columns for performance)
            df = pd.read_csv(
                csv_path,
                usecols=[id_column],
                dtype={id_column: "Int64"}
            )
            
            # Extract IDs
            ids = df[id_column].dropna().unique().astype(str).tolist()
            
            # If conversion from hadm_id to stay_id is needed
            if use_hadm_id:
                _logger.info(f"Converting hadm_id to stay_id...")
                loader = DataLoader(data_dir=data_dir)
                stay_ids = []
                for hadm_id in ids:
                    stay_id = loader._get_stay_id_from_hadm_id(hadm_id)
                    if stay_id:
                        stay_ids.append(stay_id)
                    else:
                        _logger.warning(f"No stay_id found for hadm_id {hadm_id}")
                ids = stay_ids
                _logger.info(f"Successfully converted {len(ids)} hadm_ids to stay_ids")
            
            if limit:
                ids = ids[:limit]
            
            _logger.info(f"Read {len(ids)} stay_ids from {csv_path}")
            return sorted(ids)
            
        except Exception as e:
            _logger = logging.getLogger(__name__)
            _logger.error(f"Failed to read stay_ids from CSV: {str(e)}")
            return []
    
    @staticmethod
    def inspect_csv_structure(csv_path: Union[str, Path], nrows: int = 5) -> Dict[str, Any]:
        """
        Inspect CSV file structure.

        Args:
            csv_path: CSV file path
            nrows: Number of leading rows to read for preview

        Returns:
            Dict with column names, row count, preview rows, etc.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return {"error": f"File not found: {csv_path}"}
        
        try:
            # Read first n rows
            df_preview = pd.read_csv(csv_path, nrows=nrows)
            
            # Get total row count (may take some time)
            try:
                df_count = pd.read_csv(csv_path, usecols=[0])
                total_rows = len(df_count)
            except Exception:
                total_rows = None
            
            return {
                "path": str(csv_path),
                "columns": df_preview.columns.tolist(),
                "total_rows": total_rows,
                "preview_rows": nrows,
                "preview_data": df_preview.to_dict('records'),
                "dtypes": {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
            }
        except Exception as e:
            return {"error": str(e)}


def load_sample_stay_ids(n: int = 10, data_dir: Optional[Path] = None) -> List[str]:
    """
    Convenience helper: load a sample list of stay_ids.

    Args:
        n: Number of stay_ids to return
        data_dir: Data directory

    Returns:
        List of stay_ids
    """
    loader = DataLoader(data_dir)
    return loader.list_available_stay_ids(limit=n, min_diagnosis_count=1)


# --------------------------- #
#  Self-test entry (for quick validation only)   #
# --------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DataLoader self-test entry")
    parser.add_argument("--stay-id", type=str, help="stay_id to test")
    parser.add_argument("--hadm-id", type=str, help="hadm_id to test")
    parser.add_argument("--csv-file", type=str, help="Read IDs from CSV (auto-convert hadm_id→stay_id)")
    parser.add_argument("--stay-id-column", type=str, default=None, help="stay_id column name in CSV")
    parser.add_argument("--hadm-id-column", type=str, default=None, help="hadm_id column name in CSV")
    parser.add_argument("--preview-only", action="store_true", help="Preview merged input only")
    parser.add_argument("--save-json", type=str, help="Save merged result to JSON (default: outputs/first_case.json for first case)")
    args = parser.parse_args()

    loader = DataLoader()
# python utils/data_loader.py --csv-file merged/discharge_target_test.csv --hadm-id-column hadm_id --save-json outputs/case1.json

    # Get IDs to test from CSV
    ids_to_test = []
    if args.csv_file:
        ids_to_test = DataLoader.load_stay_ids_from_csv(
            args.csv_file,
            stay_id_column=args.stay_id_column,
            hadm_id_column=args.hadm_id_column,
        )
        if not ids_to_test:
            print(f"[ERROR] Unable to read IDs from {args.csv_file}")
            exit(1)
        # Take only the first ID for quick self-test
        ids_to_test = ids_to_test[:1]
    elif args.stay_id or args.hadm_id:
        ids_to_test = [args.stay_id or args.hadm_id]
    else:
        # Default: take one available stay_id
        sample_ids = loader.list_available_stay_ids(limit=1, min_diagnosis_count=1)
        if not sample_ids:
            print("[ERROR] No sample stay_id found, please check data files")
            exit(1)
        ids_to_test = sample_ids

    for idx, sid in enumerate(ids_to_test):
        case = loader.load_case_inputs(
            stay_id=sid,
            hadm_id=args.hadm_id if args.hadm_id else None,
            drop_discharge_labels=True,
            strip_bhc_di_sections=True,
        )

        print("\n" + "=" * 80)
        print(f"Merged input preview (stay_id={case['ids']['stay_id']}, hadm_id={case['ids']['hadm_id']})")
        print("- edstays:     ", "yes" if case.get("edstays") else "no")
        print("- diagnosis:   ", len(case.get("diagnosis", [])))
        print("- triage:      ", "yes" if case.get("triage") else "no")
        print("- radiology:   ", len(case.get("radiology", [])))
        print("- discharge:   ", len(case.get("discharge", [])), "(BHC/DI label columns removed and sections stripped)")

        # Print partial content
        if case.get("triage"):
            print("\ntriage sample:")
            for k, v in list(case["triage"].items())[:8]:
                print(f"  {k}: {v}")
        if case.get("diagnosis"):
            print("\ndiagnosis first 3:")
            for i, d in enumerate(case["diagnosis"][:3], 1):
                print(f"  {i}) {d.get('icd_title')} ({d.get('icd_code')})")
        if case.get("radiology"):
            print("\nradiology first 3:")
            for idx, entry in enumerate(case["radiology"][:3], 1):
                print(f"  [{idx}]")
                for k, v in list(entry.items())[:8]:
                    print(f"    {k}: {v}")
            if len(case["radiology"]) > 3:
                print(f"  ... total {len(case['radiology'])} records")
        if case.get("discharge"):
            print("\ndischarge first 3 (text truncated):")
            for idx, entry in enumerate(case["discharge"][:3], 1):
                print(f"  [{idx}]")
                for k, v in list(entry.items())[:8]:
                    if isinstance(v, str) and len(v) > 200:
                        v = v[:200] + "..."
                    print(f"    {k}: {v}")
            if len(case["discharge"]) > 3:
                print(f"  ... total {len(case['discharge'])} records")

        # Save JSON: if not specified, default to saving first case to outputs/first_case.json
        if args.save_json or idx == 0:
            def clean(obj):
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean(x) for x in obj]
                if isinstance(obj, float) and (obj != obj):  # NaN
                    return None
                return obj

            if args.save_json:
                out_path = Path(args.save_json)
            else:
                out_dir = Path("outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "first_case.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(clean(case), f, ensure_ascii=False, indent=2)
            print(f"\n[OK] Data saved to: {out_path}")
