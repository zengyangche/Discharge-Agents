"""
Generator Agent (Expert Agents)
Uses multiple models to generate BHC and DI candidate drafts.
"""
from typing import Dict, Any, List, Optional
import logging
from utils.config import MODEL_ZOO

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    Generator agent: uses a specified model to generate BHC or DI.

    Each model acts as an independent "expert" producing candidate drafts.
    """
    
    def __init__(
        self,
        model_name: str,
        role: str,  # "bhc" or "di"
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model_name: Model name (e.g. "gpt-4o")
            role: Role ("bhc" or "di")
        """
        self.model_name = model_name
        self.role = role.lower()
        self.config = config or {}
        
        if self.role not in ["bhc", "di"]:
            raise ValueError(f"role must be 'bhc' or 'di', got: {role}")
        
        # Get model configuration
        model_config = MODEL_ZOO.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_config = model_config
        self.logger = logging.getLogger(f"{__name__}.{model_name}-{role}")
    
    def generate(
        self,
        shared_context: Dict[str, Any],
        specific_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate BHC or DI.

        Args:
            shared_context: Shared context
            specific_context: Role-specific context (BHC or DI)
            
        Returns:
            {
                "output": str,  # generated text
                "model_name": str,
                "role": str,
                "metadata": {...}
            }
        """
        try:
            prompt = self._build_prompt(shared_context, specific_context)
            system_prompt = self._get_system_prompt()
            
            # Call LLM to generate
            output = self._call_llm(prompt, system_prompt)
            
            return {
                "output": output,
                "model_name": self.model_name,
                "role": self.role,
                "metadata": {
                    "prompt_length": len(prompt),
                    "output_length": len(output),
                }
            }
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        if self.role == "bhc":
            return """You are a professional medical documentation expert specializing in writing Brief Hospital Course (BHC) in the style of MIMIC-IV discharge notes.

The BHC must include:
1. Reason for admission and chief complaint
2. Major diagnoses and treatment processes during hospitalization
3. Key examination results, imaging, and laboratory findings
4. Treatment response and changes in clinical condition
5. Status at discharge

Requirements:
- Objective, accurate, and professional; write in third person, past tense
- Use standard medical abbreviations (e.g., IV, PO, PRN, BID, EKG)
- Organize by numbered problem list when there are ≥2 distinct clinical problems; use chronological narrative for single-problem admissions
- Do not duplicate the admission chief complaint across multiple sections
- Do not include discharge medications, follow-up appointments, or patient instructions (those belong in DI)
- Base ALL content on provided data — never fabricate diagnoses, lab values, or procedures
- Write in English only"""
        else:  # di
            return """You are a professional medical documentation expert specializing in writing Discharge Instructions (DI) in the style of MIMIC-IV discharge notes.

The DI must include:
1. Reason for hospitalization and what was done (in plain language)
2. Discharge medications with drug name, dosage, frequency, and route
3. Activity restrictions and recovery recommendations
4. Dietary guidelines
5. Wound care instructions (if applicable)
6. Follow-up arrangements (provider, timeframe)
7. Warning signs and when to seek immediate medical help (at least 2–3 specific signs)

Requirements:
- Address the patient directly; begin with "Dear Mr./Ms. ___,"
- Use patient-friendly language; avoid excessive medical jargon
- Be clear, specific, and actionable — avoid vague phrases like "take care of yourself"
- Do not add medications or clinical details not present in the source data
- Close with a warm farewell (e.g., "We wish you a speedy recovery.")
- Write in English only"""
    
    def _build_prompt(
        self,
        shared_context: Dict[str, Any],
        specific_context: Dict[str, Any]
    ) -> str:
        """Build prompt"""
        if self.role == "bhc":
            return self._build_bhc_prompt(shared_context, specific_context)
        else:
            return self._build_di_prompt(shared_context, specific_context)
    
    def _build_bhc_prompt(
        self,
        shared_context: Dict[str, Any],
        specific_context: Dict[str, Any]
    ) -> str:
        """Build BHC generation prompt"""
        parts = [
            "Please generate a Brief Hospital Course (BHC) based on the following information:",
            "",
            "=== Patient Basic Information ===",
            f"Gender: {shared_context.get('patient_info', {}).get('gender', 'N/A')}",
            f"Race: {shared_context.get('patient_info', {}).get('race', 'N/A')}",
            "",
            "=== Admission Information ===",
            f"Admission Time: {shared_context.get('admission_info', {}).get('intime', 'N/A')}",
            f"Discharge Time: {shared_context.get('admission_info', {}).get('outtime', 'N/A')}",
            "",
            "=== All Diagnoses ===",
        ]
        
        for diag in specific_context.get("all_diagnoses", []):
            parts.append(f"- {diag.get('icd_title', 'N/A')} (ICD: {diag.get('icd_code', 'N/A')})")
        
        # Triage information
        triage = specific_context.get("triage_info", {})
        if triage:
            parts.extend([
                "",
                "=== Triage Information ===",
                f"Chief Complaint: {triage.get('chiefcomplaint', 'N/A')}",
                f"Temperature: {triage.get('temperature', 'N/A')}",
                f"Heart Rate: {triage.get('heartrate', 'N/A')}",
                f"Respiratory Rate: {triage.get('resprate', 'N/A')}",
                f"O2 Saturation: {triage.get('o2sat', 'N/A')}",
                f"Blood Pressure: {triage.get('sbp', 'N/A')}/{triage.get('dbp', 'N/A')}",
            ])
        
        parts.append("")
        parts.append("Write the Brief Hospital Course below. Use a numbered problem-list format if there are multiple diagnoses. Be concise, factual, and use standard medical abbreviations. Do not invent any information absent from the data above.")
        
        return "\n".join(parts)
    
    def _build_di_prompt(
        self,
        shared_context: Dict[str, Any],
        specific_context: Dict[str, Any]
    ) -> str:
        """Build DI generation prompt"""
        gender = shared_context.get('patient_info', {}).get('gender', 'N/A')
        gender_pronoun = "his" if str(gender).lower() in ("m", "male") else "her" if str(gender).lower() in ("f", "female") else "their"

        parts = [
            "Generate Discharge Instructions (DI) based on the following clinical data.",
            "",
            "=== Patient Basic Information ===",
            f"Gender: {gender} (use pronoun: {gender_pronoun})",
            "",
            "=== Primary Diagnoses ===",
        ]

        for diag in shared_context.get("primary_diagnoses", []):
            parts.append(f"- {diag.get('icd_title', 'N/A')} (ICD: {diag.get('icd_code', 'N/A')})")

        discharge_meds = specific_context.get("discharge_medications", [])
        if discharge_meds:
            parts.extend(["", "=== Discharge Medications ==="])
            for med in discharge_meds:
                if isinstance(med, dict):
                    parts.append(
                        f"- {med.get('name', 'N/A')} {med.get('dose', '')} {med.get('route', '')} {med.get('frequency', '')}".strip()
                    )
                else:
                    parts.append(f"- {med}")

        followup = specific_context.get("followup_instructions", "")
        if followup:
            parts.extend(["", "=== Follow-up Instructions ===", str(followup)])

        activity = specific_context.get("activity_restrictions", "")
        if activity:
            parts.extend(["", "=== Activity Restrictions ===", str(activity)])

        diet = specific_context.get("diet_instructions", "")
        if diet:
            parts.extend(["", "=== Dietary Instructions ===", str(diet)])

        parts.extend([
            "",
            "=== Discharge Disposition ===",
            f"{specific_context.get('disposition', 'N/A')}",
            "",
            "=== Required Output Structure ===",
            "Your Discharge Instructions MUST cover all of the following elements:",
            "1. Greeting addressed to the patient (Dear Mr./Ms. ___,)",
            "2. Brief plain-language explanation of why the patient was admitted and what was done",
            "3. Discharge medications — name, dose, frequency, and route for each",
            "4. Activity restrictions and recovery recommendations",
            "5. Dietary guidelines",
            "6. Follow-up appointments and contact information",
            "7. At least 2–3 specific warning signs requiring immediate medical attention",
            "8. Warm closing farewell",
            "",
            "Write the Discharge Instructions below. Address the patient directly. Use plain language. "
            "Do not invent medications, dosages, or follow-up details that are absent from the data above.",
        ])

        return "\n".join(parts)
    
    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """
        Call the LLM to generate text.

        Unified interface that routes to the appropriate API per model config.
        """
        provider = self.model_config["provider"]
        
        if provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif provider == "qwen":
            return self._call_qwen(prompt, system_prompt)
        elif provider == "deepseek":
            return self._call_deepseek(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            from utils.config import OPENAI_API_KEY, OPENAI_API_BASE_URL
            
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE_URL
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 2000),
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _call_qwen(self, prompt: str, system_prompt: str) -> str:
        """Call Qwen API (placeholder; implement per actual API)"""
        # TODO: Implement Qwen API call
        raise NotImplementedError("Qwen API not implemented, please implement per actual API documentation")
    
    def _call_deepseek(self, prompt: str, system_prompt: str) -> str:
        """Call DeepSeek API (placeholder; implement per actual API)"""
        # TODO: Implement DeepSeek API call
        raise NotImplementedError("DeepSeek API not implemented, please implement per actual API documentation")


class GeneratorOrchestrator:
    """
    Generation orchestrator: manages multiple expert agents and parallel draft generation.
    """
    
    def __init__(self, model_names: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_names: List of model names (e.g. ["gpt-4o", "qwen-max", "deepseek-r1"])
        """
        self.model_names = model_names
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.GeneratorOrchestrator")
    
    def generate_all(
        self,
        shared_context: Dict[str, Any],
        bhc_context: Dict[str, Any],
        di_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate all BHC and DI candidate drafts using every configured model.

        Returns:
            {
                "bhc_outputs": {
                    "model_name": {
                        "output": str,
                        "metadata": {...}
                    },
                    ...
                },
                "di_outputs": {
                    "model_name": {
                        "output": str,
                        "metadata": {...}
                    },
                    ...
                }
            }
        """
        bhc_outputs = {}
        di_outputs = {}
        
        # Generate BHC
        self.logger.info(f"Starting BHC generation (using {len(self.model_names)} models)...")
        for model_name in self.model_names:
            try:
                agent = GeneratorAgent(model_name, "bhc", self.config)
                result = agent.generate(shared_context, bhc_context)
                bhc_outputs[model_name] = result
                self.logger.info(f"BHC generation complete ({model_name})")
            except Exception as e:
                self.logger.error(f"BHC generation failed ({model_name}): {str(e)}")
                bhc_outputs[model_name] = None
        
        # Generate DI
        self.logger.info(f"Starting DI generation (using {len(self.model_names)} models)...")
        for model_name in self.model_names:
            try:
                agent = GeneratorAgent(model_name, "di", self.config)
                result = agent.generate(shared_context, di_context)
                di_outputs[model_name] = result
                self.logger.info(f"DI generation complete ({model_name})")
            except Exception as e:
                self.logger.error(f"DI generation failed ({model_name}): {str(e)}")
                di_outputs[model_name] = None
        
        return {
            "bhc_outputs": bhc_outputs,
            "di_outputs": di_outputs,
        }
