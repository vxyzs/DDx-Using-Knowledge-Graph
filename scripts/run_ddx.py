"""
run_ddx_integrated.py

Orchestrates the complete diagnostic pipeline:
1. NLU (semantic search) → fast candidate finding with confidence scores
2. Parser (LLM refinement) → structured extraction with reasoning
3. Traversal (interactive diagnosis) → adaptive Bayesian updates

This is the improved flow that connects NLU → Parser → Traversal.
"""

import pickle
import json
from functools import lru_cache
from typing import Optional, Dict, List, Any

# Imports from local modules
from core.nlu import DDxGraphNLU
from core.parser import Parser
from core.traversal import KG_Traversal


@lru_cache(maxsize=1)
def load_graph(path="./Pickle/kg.pkl"):
    """
    Loads the knowledge graph once and caches it.
    Prevents re-loading if called multiple times.
    """
    print("[Graph Loading] Initializing Knowledge Graph...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"[Graph Loading] ✓ Loaded {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def initialize_scores(G):
    """
    Efficiently initialize condition node scores to 0.0.
    Avoids repeated dictionary lookups.
    """
    condition_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == "condition"
    ]
    return dict.fromkeys(condition_nodes, 0.0)


class IntegratedDiagnosticPipeline:
    """
    Coordinates the three-stage diagnostic flow.
    
    Stage 1: NLU (nlu.py)
      - Semantic matching: encodes user text, finds similar evidence
      - Output: evidence IDs, confidence scores, negation flags
    
    Stage 2: Parser (parser.py)
      - LLM-based refinement: uses NLU findings as hints
      - Output: structured evidences + values (Pydantic)
    
    Stage 3: Traversal (traversal.py)
      - Interactive Bayesian diagnosis: applies evidence, asks questions
      - Output: ranked conditions
    """
    
    def __init__(self, G, user_input: str):
        self.G = G
        self.user_input = user_input
        
        # Initialize NLU and Parser once
        self.nlu = DDxGraphNLU(G)
        self.parser = Parser()
        
        # Results from each stage (populated during run())
        self.nlu_result = None
        self.parser_result = None
        self.traversal = None
    
    def run(self, max_traversal_steps: int = 5, top_k_conditions: int = 10):
        """
        Execute the full pipeline: NLU → Parser → Traversal.
        """
        print("\n" + "="*80)
        print("INTEGRATED DIAGNOSTIC PIPELINE")
        print("="*80)
        
        # --- STAGE 1: NLU ---
        print("\n[STAGE 1: NLU] Semantic Search for Evidence")
        print("-" * 80)
        self._run_nlu()
        
        # --- STAGE 2: PARSER ---
        print("\n[STAGE 2: PARSER] LLM Refinement with NLU Context")
        print("-" * 80)
        self._run_parser()
        
        # --- STAGE 3: TRAVERSAL ---
        print("\n[STAGE 3: TRAVERSAL] Interactive Bayesian Diagnosis")
        print("-" * 80)
        self._run_traversal(max_steps=max_traversal_steps, top_k=top_k_conditions)
        
        print("\n" + "="*80)
        print("END OF PIPELINE")
        print("="*80)
    
    def _run_nlu(self):
        """
        STAGE 1: Run NLU semantic search.
        
        Output format:
        {
            'evidences': ['E_123', 'E_201', ...],
            'values': [['V_45'], 'NO', ...],
            'raw_findings': [
                {
                    'eid': 'E_123',
                    'value': ['V_45'],
                    'negated': False,
                    'score': 0.82,
                    'match_type': 'value_match'
                },
                ...
            ],
            'compact_json': [<evidence JSON objects>]
        }
        """
        raw_findings, evidences, values = self.nlu.parse_query(self.user_input)
        compact_json = json.loads(self.nlu.retrieve(self.user_input))
        
        self.nlu_result = {
            'evidences': evidences,
            'values': values,
            'raw_findings': raw_findings,
            'compact_json': compact_json
        }
        
        # Print NLU summary
        print(f"\n📊 NLU Summary:")
        print(f"   • Matched {len(self.nlu_result['evidences'])} evidence items")
        print(f"   • Found {len(raw_findings)} raw findings")
        
        if raw_findings:
            print(f"\n   Top NLU Findings (by confidence):")
            sorted_findings = sorted(
                raw_findings,
                key=lambda x: x.get('score', 0),
                reverse=True
            )[:5]
            for i, finding in enumerate(sorted_findings, 1):
                status = "NEGATED" if finding.get('negated') else "PRESENT"
                values_str = f"(values: {finding.get('value', [])})" if finding.get('value') else ""
                print(f"      {i}. {finding['eid']:12s} │ Score: {finding.get('score', 0):.3f} │ {status} {values_str}")
    
    def _run_parser(self):
        """
        STAGE 2: Run LLM-based Parser.
        
        Takes NLU findings as context and refines them with LLM reasoning.
        """
        # Build context for Parser: compact JSON from NLU
        context = self.nlu_result['compact_json']
        
        # Call Parser with NLU context as hint
        parsed_data = self.parser.parse_query(
            text=self.user_input,
            context=context
        )
        
        self.parser_result = {
            'evidences': parsed_data[0] if parsed_data else [],
            'values': parsed_data[1] if parsed_data else []
        }
        
        # Print Parser summary
        print(f"\n🔍 Parser Summary (LLM-refined):")
        print(f"   • Extracted {len(self.parser_result['evidences'])} evidences")
        print(f"   • Assigned values: {self.parser_result['values']}")
        
        # Compare NLU vs Parser
        self._compare_nlu_vs_parser()
    
    def _compare_nlu_vs_parser(self):
        """
        Diagnostic comparison: what did NLU find vs what did Parser refine?
        """
        nlu_eids = set(self.nlu_result['evidences'])
        parser_eids = set(self.parser_result['evidences'])
        
        only_nlu = nlu_eids - parser_eids
        only_parser = parser_eids - nlu_eids
        both = nlu_eids & parser_eids
        
        print(f"\n   ⚖️  NLU vs Parser Comparison:")
        print(f"      • Both found:        {len(both):2d} evidences")
        print(f"      • Only NLU found:    {len(only_nlu):2d} evidences")
        print(f"      • Only Parser found: {len(only_parser):2d} evidences")
        
        if only_nlu:
            print(f"      • NLU-only: {', '.join(sorted(only_nlu)[:3])}" + 
                  (f" + {len(only_nlu) - 3} more" if len(only_nlu) > 3 else ""))
        if only_parser:
            print(f"      • Parser-only: {', '.join(sorted(only_parser)[:3])}" +
                  (f" + {len(only_parser) - 3} more" if len(only_parser) > 3 else ""))
    
    def _run_traversal(self, max_steps: int = 5, top_k: int = 10):
        """
        STAGE 3: Run interactive Traversal.
        
        Uses Parser output as initial state, then asks discriminating questions.
        """
        # Initialize condition scores
        scores = initialize_scores(self.G)
        
        # Create Traversal instance
        # Note: KG_Traversal expects user_input and will call parse_query internally
        # We pass parsed evidence to it via apply_initial_evidence after init
        self.traversal = KG_Traversal(
            G=self.G,
            scores=scores,
            user_input=None  # We'll feed evidence manually
        )
        
        # Apply Parser findings as initial evidence
        print(f"\n📥 Initializing Traversal with Parser findings...")
        self.traversal.apply_initial_evidence(
            self.parser_result['evidences'],
            self.parser_result['values']
        )
        
        print(f"   ✓ Applied {len(self.parser_result['evidences'])} initial evidences")
        print(f"   • Observed YES: {len(self.traversal.observed_yes)} items")
        print(f"   • Observed NO:  {len(self.traversal.observed_no)} items")
        
        # Run interactive loop
        print(f"\n🏥 Starting Interactive Questions (max {max_steps} steps)...")
        self.traversal.run(max_steps=max_steps, top_k_conditions=top_k)
    
    def print_summary(self):
        """
        Print final summary and diagnostics.
        """
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        print("\n1️⃣ NLU Results:")
        print(f"   Evidences: {len(self.nlu_result['evidences'])}")
        print(f"   Raw Findings: {len(self.nlu_result['raw_findings'])}")
        
        print("\n2️⃣ Parser Results:")
        print(f"   Evidences: {len(self.parser_result['evidences'])}")
        print(f"   Values: {len(self.parser_result['values'])}")
        
        print("\n3️⃣ Traversal Results:")
        if self.traversal:
            ranked = sorted(
                self.traversal.scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print(f"   Top 5 Conditions:")
            for cond, score in ranked:
                print(f"      • {cond:40s} score={score:.4f}")


def main():
    """
    Main entry point for the integrated pipeline.
    """
    try:
        # Load Knowledge Graph
        G = load_graph()
        
        # Get user input
        user_input = input("\n🩺 Describe your symptoms: ").strip()
        
        if not user_input:
            print("❌ No input provided. Exiting.")
            return
        
        # Create and run pipeline
        pipeline = IntegratedDiagnosticPipeline(G, user_input)
        pipeline.run(max_traversal_steps=5, top_k_conditions=10)
        
        # Print summary
        pipeline.print_summary()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Ensure './Pickle/kg.pkl' exists and contains the Knowledge Graph.")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()