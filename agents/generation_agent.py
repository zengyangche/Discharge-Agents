"""
Generation Agent: uses multiple LLMs to generate BHC and DI
"""
import re
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Generation Agent: uses a specified LLM to generate Brief Hospital Course (BHC) or Discharge Instructions (DI)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_shots: int = 3,
        examples_dir: Optional[str] = None,
        retriever=None,  # Optional[FewShotRetriever], enables dynamic top-K retrieval when provided
    ):
        super().__init__("GenerationAgent", config)
        self.llm_client = LLMClient()
        
        # Greedy decoding: temperature=0, no frequency_penalty, ensures deterministic output
        self.default_temperature = self.config.get("temperature", 0.0)
        self.default_frequency_penalty = self.config.get("frequency_penalty", 0.0)
        self.default_min_tokens = self.config.get("min_tokens", 200)
        self.default_max_tokens = self.config.get("max_tokens", 1024)

        # Few-shot configuration
        self.num_shots = num_shots
        self.examples_dir = Path(examples_dir) if examples_dir else Path("outputs/test_labels")
        # Dynamic retriever (if not None, replaces static examples with top-K retrieval at inference time)
        self.retriever = retriever

        # BHC system prompt
        self.bhc_system_prompt = """You are an expert attending physician writing clinical documentation for a hospital discharge record (MIMIC-IV style).

IMPORTANT CONTEXT: The input clinical notes have had their ACTIVE ISSUES / ASSESSMENT & PLAN / HOSPITAL COURSE section removed (to prevent data leakage). You must RECONSTRUCT each active problem by synthesizing information from the remaining data: the full HPI, ED Course summary, Pertinent Results (labs/imaging), medications given, and consult notes.

Your task: fill the slots below in order to produce a Brief Hospital Course (BHC).

SLOT 1 — Opening sentence (REQUIRED)
  Source: Shared Context → "Discharge Diagnosis: PRIMARY" list for the admission reason;
          Full HPI for comorbidities and clinical context.
  Rule: Use the PRIMARY Discharge Diagnosis (not the ICD triage code) as the admission reason.
  Format: "___ is a [sex] with [key PMH comorbidities] who presented with / was admitted for [PRIMARY Discharge Diagnosis]."
  Rule: one sentence; third person, past tense; use ___ for name/age.

SLOT 2 — Active / Acute Problems (REQUIRED — use Discharge Diagnosis as your problem list)
  STEP A: Find the "Discharge Diagnosis:" section in the notes — this lists ALL final diagnoses.
    PRIMARY diagnoses are acute issues; SECONDARY diagnoses may be active or chronic.
    You MUST write a # section for EVERY diagnosis listed under PRIMARY.
    Also write sections for SECONDARY diagnoses that were actively managed during the stay.
    If no Discharge Diagnosis section exists, identify problems from:
    - Full HPI narrative and ED Course summary ("Labs showed: X, Y, Z")
    - "Pertinent Results" — abnormal labs and imaging findings
    - Medications given during stay — each treatment implies a problem
  STEP B: For EACH problem from STEP A, write a "# [Problem Name]" header and describe:
    - diagnostic findings (specific lab values, imaging results, exam findings)
    - treatments given (medications with route/dose/duration, procedures, consults)
    - patient response and clinical trajectory
    - status or plan at discharge
  Rules:
    - Do NOT skip any PRIMARY Discharge Diagnosis — completeness is essential
    - Use specific values (e.g., "Na 128 → 132 by discharge") from Pertinent Results
    - If a diagnosis has limited supporting data, write a brief paragraph; do not omit it
    - Do NOT invent any values or facts not present in the source

SLOT 3 — Chronic Issues (REQUIRED if any chronic conditions are present)
  Source: "Past Medical History" section + cross-reference with Discharge Medications
    (every discharge medication corresponds to a chronic condition being managed)
  Format: "# [Condition]: continued/adjusted [medication name + dose if in discharge meds]"
  Rule: list ALL chronic conditions from PMH; include medications from discharge med list that correspond to them.

SLOT 4 — Transitional Issues (include ONLY if present in source)
  Source: "TRANSITIONAL ISSUES", "TRANSITIONS OF CARE", or pending lab/follow-up items in the notes
  Format: checkbox list ([ ] item); include code status and contact person if stated.

Assembly rules:
- Combine slots 1 → 2 → 3 → 4 in order; omit any slot that has no supporting source text.
- Do NOT include discharge medications or patient-facing instructions (those belong in DI).
- Do NOT invent diagnoses, lab values, medications, or procedures not present in the source.
- Use standard medical abbreviations (IV, PO, PRN, BID, EKG, CBC, etc.).
- Length: match clinical complexity — simple cases ~100–200 words; complex cases 300–600 words."""

        # DI system prompt
        self.di_system_prompt = """You are an expert attending physician writing patient-facing discharge instructions for a hospital discharge record (MIMIC-IV style).

Your task: fill the slots below in order to produce Discharge Instructions (DI).

SLOT 1 — Greeting (REQUIRED)
  Source: Patient gender from Shared Context → "gender" field (M → "Mr.", F → "Ms.", unknown → omit title)
  Format: "Dear Mr. ___," or "Dear Ms. ___,"

SLOT 2 — Why you were in the hospital (REQUIRED)
  Source: Full HPI from clinical notes — find the TRUE reason for admission
  CRITICAL: Do NOT use the ICD triage code alone. Read the HPI to find the actual admission diagnosis
    (look for "admitted for", "presenting with", the main condition driving the hospitalization).
  Format: 1–3 plain-language sentences a non-medical patient can understand.
  Example: "You were admitted because your blood sodium level was dangerously low (hyponatremia)..."

SLOT 3 — What happened / what we did (REQUIRED)
  Source: Brief Hospital Course (already provided) → all active problem summaries
  Format: use the narrative style matching MIMIC discharge instructions:
    For complex cases: bullet points "- You were found to have [condition]. [Treatment and outcome.]"
    For simple cases: 1–2 short paragraphs
  Rule: cover ALL key problems from the BHC; translate medical terms to plain language.

SLOT 4 — Discharge medications (include ONLY if a "Discharge Medications:" list exists in the source)
  Source: DI-Specific Source Data → "Discharge Medications:" list
  Format: numbered list, copy name/dose/frequency EXACTLY as written in the source.
  Rule: include ALL medications listed — do NOT omit any; do NOT add medications not in the list.

SLOT 5 — Activity / special instructions (include ONLY if explicitly stated)
  Source: DI-Specific Source Data or BHC → WEIGHTBEARING STATUS, wound care, dietary restrictions
  Format: copy and simplify as needed.

SLOT 6 — Follow-up (include ONLY if data is available)
  Source: BHC → TRANSITIONAL ISSUES ([ ] items); DI-Specific → Followup Instructions
  Format: brief sentences "Please follow up with [provider] for [reason]."

SLOT 7 — Warning signs (REQUIRED)
  Source: Main diagnoses from the BHC → generate relevant symptoms
  Format: "Please return to the emergency room or call your doctor immediately if you experience: [symptom], [symptom], ..."
  Rule: symptoms must be specific to this patient's conditions.

SLOT 8 — Closing (REQUIRED)
  Format: "We wish you the best! Sincerely, Your ___ Team"
  (Match the tone of the examples — brief and warm.)

Assembly rules:
- Write slots 1 → 8 in order; omit slots with no supporting data.
- Use plain, patient-friendly language throughout; avoid medical jargon.
- Do NOT reproduce detailed lab values (those belong in BHC).
- Do NOT invent medications, dosages, or follow-up appointments not in the source.
- LENGTH CONSTRAINT: Target 100–200 words for simple cases (1–2 problems); up to 300 words for complex cases (3+ problems). Be concise — do not add extra explanation beyond what the patient needs to act on. If a slot has no new information to add, omit it rather than padding."""
        
        # ── Built-in BHC examples (3 entries, from real de-identified discharge records) ──────────────────────────
        self._builtin_bhc_list: List[str] = [
            # Example 1: CAD/post-coronary stent chest pain (single problem, narrative style)
            "Mr. ___ is a ___ year old man with history of CAD, LAD stent placed ___, "
            "hypertension, newly diagnosed metastatic prostate cancer who presented with chest "
            "pain several days after discharge from LAD stenting. He underwent cardiac cath on "
            "___ which was unchanged from prior. He was discharged with plan for outpatient "
            "stress test.",

            # Example 2: DKA + multiple comorbidities (multi-problem numbered list format)
            "___ h/o DM I who was transferred from ___ to ___ ICU for DKA ___. He was downgraded to the floor ___.\n\n"
            "1. DKA\n"
            "-On admission to ___ pH 7.19 w/ bicarb 9 and transferred to ___ IVU treated with IV fluids and IV insulin "
            "with resolution of DKA successfully transitioned to SC insulin and transferred to the floor.  ___ followed "
            "during the admission adjusting insulin discharged on lantus 20Units daily and Humalog 8units TID. Education "
            "was provided regarding cause of DKA due to skipping insulin, which patient didn't know.  Also discussed that "
            "in setting of HbA1C 9.3% ___ he has likely been suboptimally controlled in the past few months. He will "
            "discuss referral to endocrinology with his PCP vs his PCP to continue to manage.\n\n"
            "2. ___ h/o CKD\n"
            "-Likely prerenal in setting of DKA and volume depletion. Patient notes h/o CKD but unknown baseline. "
            "Creatinine 2.2 on admission at ___ and 1.4 at last visit in  ___. With IV fluids creatinine improved to "
            "1.2 at discharge.\n\n"
            "3. ETOH abuse\n"
            "-___ drinks/day without previous history of withdrawal.  He received phenobarbital ___ and ___ but refused "
            "further taper without evidence of withdrawal. Social work met with him where patient admitted to strong "
            "family history of EtOH abuse and concern for alcohol abuse himself.  He is considering reengaging in "
            "outpatient counseling.\n\n"
            "4. Insomnia\n-Benadryl at request with ramelteon.\n\n"
            "5. Chest pain, dyspnea, palpitations\n"
            "-Likley in setting of DKA with increased energy drink intake.  EKG w/out ischemic changes and troponin "
            "peaked at 0.02.  Symptoms since resolved with resolution of DKA.\n\n"
            "6. Pancytopenia (Macrocytic Anemia)\n-Likely in setting of EtOH abuse.\n\n"
            "7. Transaminitis\n-Likely from EtOH use. Monitor.\n\n"
            "CHRONIC MEDICAL PROBLEMS\n1. Hypothyroidism: continue Levothyroxine",

            # Example 3: Small bowel obstruction (multi-system organ-based format)
            "Mr. ___ presented to the emergency department on ___ with reports of nausea, vomiting, and abdominal pain. "
            "He underwent a CT scan that was indicative of a small bowel obstruction. The patient was admitted to the "
            "colorectal surgery team for conservative management.\n\n"
            "Neuro: Pain was well controlled on Tylenol and oxycodone for breakthrough pain.\n\n"
            "CV: The patient remained stable from a cardiovascular standpoint; vital signs were routinely monitored.\n\n"
            "Pulm: The patient remained stable from a pulmonary standpoint; oxygen saturation was routinely monitored. "
            "Had good pulmonary toileting, as early ambulation and incentive spirometry were encouraged throughout "
            "hospitalization.\n\n"
            "GI: The patient was initially kept NPO for bowel rest. Once his abdominal exam improved and the patient had "
            "ostomy function, he was slowly advanced to a regular diet. At time of discharge, he was tolerating a regular "
            "diet and his intake and output was closely monitored.\n\n"
            "GU: The patient had good urine output throughout hospitalization. He will be discharged with a Foley catheter "
            "due to history of urinary retention and inability to pass a voiding trial. He will follow up with urology as "
            "an outpatient to have his foley catheter removed.\n\n"
            "ID: The patient was closely monitored for signs and symptoms of infection and fever of which there was none.\n\n"
            "Heme: The patient had blood levels checked daily during their hospital course to monitor for signs of "
            "bleeding. The patient remained on his prophylactic lovenox and ___ dyne boots were used during this stay. "
            "He was encouraged to get up and ambulate as early as possible. He will continue his prophylactic course of "
            "lovenox at home.\n\n"
            "On ___, the patient was discharged to home. At discharge, he was tolerating a regular diet, passing flatus, "
            "voiding, and ambulating independently. He will follow-up in the clinic in ___ weeks. This information was "
            "communicated to the patient directly prior to discharge.",
        ]

        # ── Built-in DI examples (3 entries) ────────────────────────────────────────────────
        self._builtin_di_list: List[str] = [
            # Example 1: PE/DVT (anticoagulation therapy)
            "Dear Ms. ___,\n\n"
            "You were transferred to ___ for evaluation of your shortness of breath. You were found to have a large "
            "clot in your lungs that likely came from the clot in your leg. You underwent a procedure called "
            "catheter-directed thrombolysis which put clot-busting medication directly into the clots. You were also "
            "started on systemic anticoagulation to prevent the formation of new clots. There was concern that you may "
            "have had an autoimmune reaction to heparin, the first anti-clotting medication you were on. You were "
            "switched to a different anti-clotting medication called fondaparinux; you should continue injecting "
            "yourself with this medication once a day. While treating your clots, your heart went into an abnormal "
            "rhythm. Please continue to take the metoprolol daily to control the rhythm. In addition, you were noted "
            "to have some extra fluid on exam; you were started on 20mg lasix that you should take daily. Please "
            "follow up with Dr. ___ your pulmonary hypertension and hematology to discuss your anti-clotting "
            "medications.\n\nThank you for letting us be a part of your care!\n- Your ___ Team",

            # Example 2: DKA
            "Mr. ___,\n\n"
            "You were admitted to the ICU with Diabetic Ketoacidosis (DKA) and treated with aggressive IV fluids and "
            "IV insulin with improvement.  DKA also causes dehydration, which caused to you have kidney damage that "
            "also improved with IV fluids.\n\n"
            "DKA developed due to not taking your insulin properly. You were seen by the endocrinology doctors (___) "
            "during your admission who adjusted your insulin.  Please take Lantus 20Units daily and Humalog 8Units "
            "with meals (breakfast, lunch, dinner).  You can adjust the timing of the insulin to fit your work "
            "schedule.\n\n"
            "Please consider following up with counselors for alcohol use.  You are high risk of alcohol abuse based "
            "on your family history. Alcohol also has a lot of carbohydrates and will make your diabetes more "
            "difficult to control.\n\nIt was a pleasure taking care of you.\n-Your ___ team",

            # Example 3: Small bowel obstruction
            "Dear Mr. ___,\n\n"
            "You were admitted to the hospital for a small bowel obstruction. You were given bowel rest, intravenous "
            "fluids, and pain medication. Your obstruction has subsequently resolved after conservative management. "
            "You are tolerating a regular diet, passing gas and your pain is controlled with pain medications by "
            "mouth.\n\n"
            "If you have any of the following symptoms please call the office for advice or go to the emergency room "
            "if severe: increasing abdominal distension, increasing abdominal pain, nausea, vomiting, inability to "
            "tolerate food or liquids, prolonged loose stool, or extended constipation.\n\n"
            "You will be discharged home on Lovenox injections to prevent blood clots after surgery. You will take "
            "this for 30 days after your surgery date, please finish the entire prescription. This will be given once "
            "daily. Please follow all nursing teaching instruction given by the nursing staff. Please monitor for any "
            "signs of bleeding: fast heart rate, bloody bowel movements, abdominal pain, bruising, feeling faint or "
            "weak. If you have any of these symptoms please call our office or seek medical attention. Avoid any "
            "contact activity while taking Lovenox. Please take extra caution to avoid falling.\n\n"
            "Thank you for allowing us to participate in your care, we wish you all the best!",
        ]

        # Load all available examples (built-in + external files), cached for prompt construction
        self._all_bhc_examples, self._all_di_examples = self._load_all_examples()

        # Cache final prompt segments based on num_shots (empty string indicates 0-shot)
        self.bhc_examples = self._build_examples_text("bhc", num_shots)
        self.di_examples = self._build_examples_text("di", num_shots)
    
    # ── Few-shot example loading and construction ────────────────────────────────────────────────

    def _load_all_examples(self):
        """Merge built-in examples with examples loaded from external directory; return (bhc_list, di_list)."""
        bhc_list = list(self._builtin_bhc_list)
        di_list = list(self._builtin_di_list)

        # Only attempt to load from files when more examples are needed
        if self.num_shots > len(bhc_list) and self.examples_dir.exists():
            needed = self.num_shots - len(bhc_list) + 5  # Load a few extra as buffer
            extra_bhc, extra_di = self._load_examples_from_dir(needed)
            bhc_list.extend(extra_bhc)
            di_list.extend(extra_di)

        return bhc_list, di_list

    def _load_examples_from_dir(self, max_count: int = 20):
        """Read label files from examples_dir and extract BHC/DI example pairs."""
        bhc_out: List[str] = []
        di_out: List[str] = []

        files = sorted(self.examples_dir.glob("case_*.txt"))
        rng = random.Random(42)
        rng.shuffle(files)  # Fixed seed for reproducibility

        for fp in files:
            if len(bhc_out) >= max_count:
                break
            try:
                content = fp.read_text(encoding="utf-8")
                bhc_m = re.search(
                    r'Brief Hospital Course.*?={80}\s*(.*?)\s*={80}',
                    content, re.DOTALL | re.IGNORECASE
                )
                di_m = re.search(
                    r'Discharge Instructions.*?={80}\s*(.*?)(?:\s*={80}|\Z)',
                    content, re.DOTALL | re.IGNORECASE
                )
                if bhc_m and di_m:
                    bhc = bhc_m.group(1).strip()
                    di = di_m.group(1).strip()
                    if len(bhc.split()) >= 50 and len(di.split()) >= 30:
                        bhc_out.append(bhc)
                        di_out.append(di)
            except Exception:
                continue

        return bhc_out, di_out

    def _build_examples_text(self, section: str, n: int) -> str:
        """
        Build an n-shot example text string.
        section: "bhc" or "di"
        Returns empty string when n == 0.
        Cycles through the pool if fewer than n examples are available.
        """
        if n == 0:
            return ""

        pool = self._all_bhc_examples if section == "bhc" else self._all_di_examples
        if not pool:
            return ""

        label = "Brief Hospital Course" if section == "bhc" else "Discharge Instruction"
        selected = [pool[i % len(pool)] for i in range(n)]

        parts = []
        for i, example in enumerate(selected, 1):
            parts.append(f"{label} Example {i} start:\n{example}\n{label} Example {i} end.")
        return "\n\n".join(parts)

    def _format_retrieved_examples(self, retrieved: list, section: str) -> str:
        """
        Format the return value of FewShotRetriever.retrieve() into a prompt fragment.

        Args:
            retrieved: list of {"bhc": str, "di": str, "score": float, ...}
            section:   "bhc" or "di"

        Returns:
            Formatted example text string (same format as _build_examples_text output).
        """
        if not retrieved:
            return ""
        label = "Brief Hospital Course" if section == "bhc" else "Discharge Instruction"
        parts = []
        for i, ex in enumerate(retrieved, 1):
            content = ex["bhc"] if section == "bhc" else ex["di"]
            parts.append(f"{label} Example {i} start:\n{content}\n{label} Example {i} end.")
        return "\n\n".join(parts)

    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Implements the abstract method process from BaseAgent.
        
        Args:
            **kwargs: contains the following parameters:
                - shared_context: shared context
                - bhc_specific_context: BHC-specific context
                - di_specific_context: DI-specific context
                - model_names: list of model names (optional, defaults to models in config)
                - other generation parameters (temperature, max_tokens, etc.)
        
        Returns:
            generation result dict
        """
        shared_context = kwargs.get("shared_context", "")
        bhc_specific_context = kwargs.get("bhc_specific_context", "")
        di_specific_context = kwargs.get("di_specific_context", "")
        model_names = kwargs.get("model_names", self.config.get("default_models", ["gpt-4o"]))
        
        # Extract generation parameters
        gen_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ["shared_context", "bhc_specific_context", 
                                  "di_specific_context", "model_names"]}
        
        return self.batch_generate(
            shared_context=shared_context,
            bhc_specific_context=bhc_specific_context,
            di_specific_context=di_specific_context,
            model_names=model_names,
            **gen_kwargs
        )
    
    def generate_bhc(
        self,
        shared_context: str,
        bhc_specific_context: str,
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Brief Hospital Course.
        
        Args:
            shared_context: shared context
            bhc_specific_context: BHC-specific context
            model_name: name of the model to use
            **kwargs: additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            {
                "content": str,  # generated BHC text
                "model": str,    # model used
                "metadata": dict # metadata
            }
        """
        try:
            self.log(f"Generating BHC with {model_name}...")
            
            # ── Determine few-shot example source ──────────────────────────────────────
            if self.num_shots > 0 and self.retriever is not None:
                query_text = (shared_context + "\n" + bhc_specific_context).strip()
                retrieved = self.retriever.retrieve(query_text, k=self.num_shots)
                bhc_examples_text = self._format_retrieved_examples(retrieved, "bhc")
                self.log(
                    f"[RAG] BHC retrieved top-{len(retrieved)} examples, "
                    f"similarity: {retrieved[0]['score']:.3f} ~ {retrieved[-1]['score']:.3f}"
                    if retrieved else "[RAG] BHC retrieval returned no results, falling back to 0-shot"
                )
            else:
                bhc_examples_text = self.bhc_examples

            if self.num_shots > 0 and bhc_examples_text:
                example_block = (
                    "Below are example Brief Hospital Courses showing the expected slot structure and style.\n\n"
                    f"{bhc_examples_text}\n\n---\n\n"
                )
            else:
                example_block = ""

            prompt = f"""{example_block}Reconstruct the Brief Hospital Course from the raw clinical data below.

⚠ CRITICAL INSTRUCTIONS BEFORE YOU START ⚠
1. The ACTIVE ISSUES / ASSESSMENT & PLAN section has been REMOVED from these notes.
2. STEP 0 — Find the "Discharge Diagnosis:" section in the notes (usually near the bottom, before "Discharge Condition:"). It lists ALL final diagnoses (PRIMARY and SECONDARY). Use these as your problem list for SLOT 2.
3. STEP 0b — The [Shared Context] shows the TRIAGE complaint code only; it is NOT the final diagnosis. Use the "Discharge Diagnosis" section and the full HPI to determine the true admission reason for SLOT 1.

=== INPUT DATA ===

[Shared Context — includes ED triage code AND the final "Discharge Diagnosis:" list]
{shared_context}

[Clinical Notes — full HPI, PMH, ED Course, Pertinent Results, Discharge Medications]
(Note: ACTIVE ISSUES / HOSPITAL COURSE / Assessment & Plan section has been removed)
{bhc_specific_context}

=== OUTPUT ===
Brief Hospital Course:"""
            
            # Call LLM (using medical text generation optimized parameters from config)
            temperature = kwargs.get("temperature", self.default_temperature)
            min_tokens = kwargs.get("min_tokens", self.default_min_tokens)
            max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            frequency_penalty = kwargs.get("frequency_penalty", self.default_frequency_penalty)
            
            content = self.llm_client.generate(
                model_name=model_name,
                prompt=prompt,
                system_prompt=self.bhc_system_prompt,
                temperature=temperature,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
            )
            
            self.log(f"BHC generation complete, length: {len(content)} characters")
            
            return {
                "content": content.strip(),
                "model": model_name,
                "metadata": {
                    "type": "BHC",
                    "temperature": temperature,
                    "min_tokens": min_tokens,
                    "max_tokens": max_tokens
                }
            }
            
        except Exception as e:
            self.log(f"Error generating BHC: {str(e)}", "ERROR")
            return {
                "content": f"[Error: {str(e)}]",
                "model": model_name,
                "metadata": {"type": "BHC", "error": str(e)}
            }
    
    def generate_di(
        self,
        shared_context: str,
        di_specific_context: str,
        model_name: str,
        generated_bhc: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Discharge Instructions.
        
        Args:
            shared_context: shared context
            di_specific_context: DI-specific context
            model_name: name of the model to use
            generated_bhc: already-generated BHC content (used for cascaded generation)
            **kwargs: additional parameters
            
        Returns:
            {
                "content": str,  # generated DI text
                "model": str,    # model used
                "metadata": dict # metadata
            }
        """
        try:
            self.log(f"Generating DI with {model_name}...")
            
            # ── Determine DI few-shot example source ────────────────────────────────────
            if self.num_shots > 0 and self.retriever is not None:
                query_text = (shared_context + "\n" + di_specific_context).strip()
                retrieved = self.retriever.retrieve(query_text, k=self.num_shots)
                di_examples_text = self._format_retrieved_examples(retrieved, "di")
                self.log(
                    f"[RAG] DI retrieved top-{len(retrieved)} examples, "
                    f"similarity: {retrieved[0]['score']:.3f} ~ {retrieved[-1]['score']:.3f}"
                    if retrieved else "[RAG] DI retrieval returned no results, falling back to 0-shot"
                )
            else:
                di_examples_text = self.di_examples

            if self.num_shots > 0 and di_examples_text:
                example_block = (
                    "Below are example Discharge Instructions showing the expected slot structure and tone.\n\n"
                    f"{di_examples_text}\n\n---\n\n"
                )
            else:
                example_block = ""

            bhc_block = (
                f"[Brief Hospital Course — already written; use as source for SLOT 2 and SLOT 3]\n{generated_bhc}"
                if generated_bhc else
                "[Brief Hospital Course — not available; derive SLOT 2 and SLOT 3 from clinical data below]"
            )

            prompt = f"""{example_block}Fill the DI slots using the data provided below.

=== INPUT DATA ===

[Shared Context — patient identifiers, gender, diagnosis codes, triage chief complaint]
{shared_context}

{bhc_block}

[DI-Specific Source Data — Discharge Medications, Discharge Disposition, Discharge Diagnosis,
 Discharge Condition, Followup Instructions, TRANSITIONAL ISSUES, Medications on Admission]
{di_specific_context}

=== OUTPUT ===
Discharge Instructions:"""
            
            # Call LLM (using medical text generation optimized parameters from config)
            temperature = kwargs.get("temperature", self.default_temperature)
            min_tokens = kwargs.get("min_tokens", self.default_min_tokens)
            max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            frequency_penalty = kwargs.get("frequency_penalty", self.default_frequency_penalty)
            
            content = self.llm_client.generate(
                model_name=model_name,
                prompt=prompt,
                system_prompt=self.di_system_prompt,
                temperature=temperature,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
            )
            
            self.log(f"DI generation complete, length: {len(content)} characters")
            
            return {
                "content": content.strip(),
                "model": model_name,
                "metadata": {
                    "type": "DI",
                    "temperature": temperature,
                    "min_tokens": min_tokens,
                    "max_tokens": max_tokens
                }
            }
            
        except Exception as e:
            self.log(f"Error generating DI: {str(e)}", "ERROR")
            return {
                "content": f"[Error: {str(e)}]",
                "model": model_name,
                "metadata": {"type": "DI", "error": str(e)}
            }
    
    def batch_generate(
        self,
        shared_context: str,
        bhc_specific_context: str,
        di_specific_context: str,
        model_names: List[str],
        show_progress: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Batch-generate BHC and DI using multiple models.
        
        Args:
            shared_context: shared context
            bhc_specific_context: BHC-specific context
            di_specific_context: DI-specific context
            model_names: list of model names
            show_progress: whether to show a progress bar
            **kwargs: additional parameters
            
        Returns:
            {
                "bhc_results": {model_name: result_dict, ...},
                "di_results": {model_name: result_dict, ...}
            }
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # If tqdm is not installed, use simple progress display
            tqdm = None
        
        bhc_results = {}
        di_results = {}
        
        total_tasks = len(model_names) * 2  # Each model generates BHC and DI
        current_task = 0  # Current task counter
        
        # Create progress bar
        if show_progress and tqdm:
            pbar = tqdm(total=total_tasks, desc="Generation progress", unit="task", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        elif show_progress:
            # If tqdm is not available, use simple text progress
            print(f"\nStarting generation, {total_tasks} tasks total...")
            pbar = None
        else:
            pbar = None
        
        try:
            for idx, model_name in enumerate(model_names):
                try:
                    # Step 1: Generate BHC first
                    if pbar:
                        pbar.set_description(f"Generating BHC ({model_name})")
                    elif show_progress:
                        current_task += 1
                        print(f"  [{current_task}/{total_tasks}] Generating BHC ({model_name})...")
                    bhc_result = self.generate_bhc(
                        shared_context, bhc_specific_context, model_name, **kwargs
                    )
                    bhc_results[model_name] = bhc_result
                    if pbar:
                        pbar.update(1)
                    
                    # Step 2: Use generated BHC to generate DI (cascaded generation)
                    generated_bhc_content = bhc_result.get('content', '')
                    if pbar:
                        pbar.set_description(f"Generating DI ({model_name}) - using generated BHC")
                    elif show_progress:
                        current_task += 1
                        print(f"  [{current_task}/{total_tasks}] Generating DI ({model_name}) - using generated BHC...")
                    di_result = self.generate_di(
                        shared_context, 
                        di_specific_context, 
                        model_name, 
                        generated_bhc=generated_bhc_content,  # Pass in the generated BHC
                        **kwargs
                    )
                    di_results[model_name] = di_result
                    if pbar:
                        pbar.update(1)
                    
                except Exception as e:
                    self.log(f"Batch generation failed for model {model_name}: {str(e)}", "ERROR")
                    bhc_results[model_name] = {"content": f"[Error: {str(e)}]", "model": model_name}
                    di_results[model_name] = {"content": f"[Error: {str(e)}]", "model": model_name}
                    if pbar:
                        pbar.update(2)  # Skip both tasks for this model
                    elif show_progress:
                        current_task += 2
                        print(f"  [{current_task}/{total_tasks}] Model {model_name} generation failed")
                
                # Check if generation result is empty (special handling for certain models)
                if model_name.startswith("gemini"):
                    if bhc_results[model_name].get("content", "").strip() == "" or len(bhc_results[model_name].get("content", "").strip()) == 0:
                        self.log(f"Warning: {model_name} generated empty BHC, possibly an API response format issue", "WARNING")
                        bhc_results[model_name]["content"] = f"[Error: {model_name} returned empty content, check API response format]"
                    if di_results[model_name].get("content", "").strip() == "" or len(di_results[model_name].get("content", "").strip()) == 0:
                        self.log(f"Warning: {model_name} generated empty DI, possibly an API response format issue", "WARNING")
                        di_results[model_name]["content"] = f"[Error: {model_name} returned empty content, check API response format]"
        finally:
            if pbar:
                pbar.close()
        
        return {
            "bhc_results": bhc_results,
            "di_results": di_results
        }


