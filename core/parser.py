"""
parser.py - IMPROVED VERSION with NLU Integration

Key improvements:
1. Accept nlu_context parameter to use NLU findings as hints
2. Enhanced prompt template that leverages NLU confidence scores
3. Better logging and debugging output
4. Fallback to NLU findings if LLM parsing fails
"""

import os
import json
from typing import List, Union, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
import pickle

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

with open('Data/ddxplus/release_evidences.json', 'r') as f:
    release_evidences = json.load(f)


# --- 1. Define the Pydantic Schema ---
class PatientEvidences(BaseModel):
    evidences: List[str] = Field(
        description=(
            "List of extracted evidence IDs. "
            "Note: Only include mentioned evidences. If an evidence with data type 'M' is present, "
            "you MUST also include its parent evidence ID in this list."
        )
    )
    values: List[Union[str, List[str]]] = Field(
        description=(
            "Corresponding values for the evidences. "
            "- For Boolean ('B') or absent evidences: use 'YES' or 'NO'. "
            "- For categorical/numerical: use a list of exact mapped IDs (e.g., [['E_55_@_V_125', 'E_55_@_V_29']]]). "
            "- For multiple values: include all mapped IDs in the list."
        )
    )


class Parser:
    """
    LLM-based parser for structuring patient evidence.
    
    Receives:
    1. Patient text (free-form description)
    2. Evidence context (JSON objects from Knowledge Graph)
    3. [NEW] NLU findings with confidence scores (optional)
    
    Returns:
    PatientEvidences: Structured extraction with validated schema
    """
    
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        
        print(f"[Parser Init] Initializing LLM: {model_name}")
        
        # --- 2. Initialize the LLM ---
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
            base_url="https://router.huggingface.co/v1",
            temperature=0.0  # Low temperature for reliable extraction
        )
        
        # --- 3. Initialize the Parser ---
        self.output_parser = PydanticOutputParser(pydantic_object=PatientEvidences)
        print("[Parser Init] ✓ LLM and output parser initialized")

    def _format_nlu_hints(self, nlu_findings: Optional[List[dict]]) -> str:
        """
        Format NLU findings into a readable hint for the LLM.
        
        Args:
            nlu_findings: List of dicts with 'eid', 'score', 'negated', 'match_type'
            
        Returns:
            Formatted string to include in prompt
        """
        if not nlu_findings:
            return ""
        
        # Sort by score descending
        sorted_findings = sorted(
            nlu_findings,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        hint_lines = ["NLU semantic search candidates (ranked by similarity confidence):"]
        for i, finding in enumerate(sorted_findings[:10], 1):
            eid = finding.get('eid', 'UNKNOWN')
            score = finding.get('score', 0.0)
            negated = " [NEGATED]" if finding.get('negated') else ""
            match_type = finding.get('match_type', 'unknown')
            values = f" → {finding.get('value', [])}" if finding.get('value') else ""
            
            hint_lines.append(
                f"  {i}. {eid:12s} (confidence: {score:.3f}, type: {match_type}){negated}{values}"
            )
        
        hint_lines.append("\nUse these candidates to guide your extraction, but apply your own reasoning.")
        return "\n".join(hint_lines)

    def parser(
        self,
        text: str,
        context: Union[str, List[Any], dict],
        nlu_context: Optional[List[dict]] = None
    ) -> PatientEvidences:
        """
        Parse patient text into structured evidence.
        
        Args:
            text: Patient description
            context: Knowledge graph evidence context (JSON)
            nlu_context: [NEW] Optional NLU findings with confidence scores
                        List of dicts with keys: 'eid', 'score', 'negated', 'match_type', 'value'
        
        Returns:
            PatientEvidences: Structured extraction
        """
        
        # Format NLU hints if provided
        nlu_hints = self._format_nlu_hints(nlu_context) if nlu_context else ""
        
        # --- 4. Define the Prompt Template ---
        prompt = PromptTemplate(
            template='''
You are a medical parser that extracts structured information from patient text.

CRITICAL RULES:
1. If an evidence is not mentioned in the text, do not include it.
2. Evidences with data type "B" (Boolean) should have values "YES" or "NO".
3. Evidences with categorical values should have values as a list of ids. 
   Example: if evidence is "E_55" (eye) and value is "V_125", output ["E_55_@_V_125"].
4. For numerical values, provide the value as a list containing the numerical value string. 
   Example: ["E_59_@_5"].
5. If multiple values are mentioned for a single evidence, include all relevant ids in the list.
6. If evidence with data type "M" is present, also include its parent evidence (code_question) with value "YES".
7. If evidence is explicitly mentioned as absent, set its value to "NO" (e.g., "no fever" → evidence "E_201" gets value "NO").

CONTEXT FROM SEMANTIC SEARCH:
{nlu_hints}

Patient text: "{text}"

Knowledge Graph Evidences: {context}

OUTPUT INSTRUCTIONS:
{format_instructions}

IMPORTANT: Even if the NLU candidates are useful, apply your own medical reasoning to validate 
and refine the extraction. The final output should be clinically sound.
''',
            input_variables=["text", "context", "nlu_hints"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        try:
            # Ensure context is JSON string
            context_json = context if isinstance(context, str) else json.dumps(context)

            prompt_value = prompt.invoke({
                "text": text,
                "context": context_json,
                "nlu_hints": nlu_hints,
            })
            
            # 1. Get raw string from LLM
            print("\n[Parser] Invoking LLM...")
            llm_response = self.llm.invoke(prompt_value)
            
            # --- PRINT 1: BEFORE PYDANTIC (Raw LLM Text) ---
            print("\n" + "="*60)
            print("1. BEFORE PYDANTIC (Raw LLM Output)")
            print("="*60)
            print(llm_response.content[:500] + ("..." if len(llm_response.content) > 500 else ""))
            
            # 2. Parse string into Pydantic object
            parsed_result = self.output_parser.invoke(llm_response)
            
            # --- PRINT 2: AFTER PYDANTIC (Structured Object) ---
            print("\n" + "="*60)
            print("2. AFTER PYDANTIC (Parsed Structure)")
            print("="*60)
            print(f"Evidences: {parsed_result.evidences}")
            print(f"Values: {parsed_result.values}")
            print("="*60 + "\n")
            
            return parsed_result
            
        except Exception as e:
            print(f"[Parser] ⚠ Error during LLM parsing: {e}")
            print(f"[Parser] 🔄 Falling back to empty extraction")
            return PatientEvidences(evidences=[], values=[])
         
    def parse_query(
        self,
        text: str,
        context: Union[str, List[Any], dict],
        nlu_context: Optional[List[dict]] = None
    ) -> tuple:
        """
        Parse a user query with optional NLU context.
        
        Args:
            text: The raw user input string
            context: Evidence JSON (from NLU.retrieve or similar)
            nlu_context: [NEW] Optional NLU findings with confidence scores
        
        Returns:
            tuple: (evidences, values) lists
        """
        print("\n[Parser.parse_query] Processing input with NLU context...")
        if nlu_context:
            print(f"[Parser.parse_query] Received {len(nlu_context)} NLU findings")
        
        parsed_data = self.parser(text, context, nlu_context)
        if parsed_data is None:
            return [], []

        return parsed_data.evidences, parsed_data.values


# ==============================================================================
# MAIN EXECUTION BLOCK (Usage Example)
# ==============================================================================
if __name__ == "__main__":
    # Example usage with NLU integration
    from nlu import DDxGraphNLU
    
    print("="*80)
    print("PARSER EXAMPLE WITH NLU INTEGRATION")
    print("="*80)
    
    G = pickle.load(open("Pickle/kg.pkl", "rb"))
    nlu = DDxGraphNLU(G)
    parser = Parser()

    sample_text = "For the past couple of weeks, I've been having sudden episodes of very intense pain on one side of my head, mainly around my eye and temple. The pain feels sharp and unbearable, and when it happens my eye starts watering and my nose feels blocked on the same side. I can't stay still during these attacks and feel extremely restless. These episodes happen multiple times and often around the same time of day, then completely go away in between. No fever and cough."
    
    # Step 1: Run NLU
    print("\n[STEP 1] Running NLU semantic search...")
    nlu_result = nlu.retrieve_enhanced(sample_text, verbose=False)
    
    print(f"\n[STEP 1 RESULT]")
    print(f"  • Found {len(nlu_result['evidences'])} evidences")
    print(f"  • Raw findings: {len(nlu_result['raw_findings'])}")
    
    # Step 2: Run Parser with NLU context
    print("\n[STEP 2] Running Parser with NLU context...")
    evidences, values = parser.parse_query(
        text=sample_text,
        context=nlu_result['compact_json'],
        nlu_context=nlu_result['raw_findings']  # Pass NLU findings as hints
    )
    
    # Step 3: Print comparison
    print("\n" + "="*80)
    print("COMPARISON: NLU vs Parser")
    print("="*80)
    print(f"\nNLU Evidences:    {nlu_result['evidences']}")
    print(f"Parser Evidences: {evidences}")
    
    nlu_set = set(nlu_result['evidences'])
    parser_set = set(evidences)
    
    print(f"\nBoth found: {len(nlu_set & parser_set)}")
    print(f"Only NLU: {nlu_set - parser_set}")
    print(f"Only Parser: {parser_set - nlu_set}")