import json
import os
import pickle
from typing import Any, List, Tuple, Union

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from core.interfaces import BaseSymptomParser
from core.nlu import DDxGraphNLU
from core.config import load_config

config = load_config()
DEFAULT_MODEL_NAME = config["parser"]["model_name"]

load_dotenv()

with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
    release_evidences = json.load(f)


class PatientEvidences(BaseModel):
    """
    Pydantic model representing structured patient evidence extraction.
    """
    evidences: List[str] = Field(
        description=(
            "List of extracted evidence IDs. Only include mentioned "
            "evidences. If an evidence with data type 'M' is present, "
            "its parent evidence ID must also be included."
        )
    )
    values: List[Union[str, List[str]]] = Field(
        description=(
            "Corresponding values for the evidences. Boolean/absent is "
            "'YES'/'NO'. Categorical/numerical is a list of exact "
            "mapped IDs (e.g., [['E_55_@_V_125']])."
        )
    )


class Parser(BaseSymptomParser):
    """
    LLM-based medical dialogue parser that extracts structured findings using
    Pydantic templates.
    """

    def __init__(self, model_name=None):
        """
        Initialize the LLM parser.

        Args:
            model_name (str, optional): LLM model identifier.
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
            base_url="https://router.huggingface.co/v1",
            temperature=0.0
        )
        self.output_parser = PydanticOutputParser(
            pydantic_object=PatientEvidences
        )

    def parser(
        self, text: str, context: Union[str, List[Any], dict]
    ) -> PatientEvidences:
        """
        Invoke LLM to extract structured patient symptoms schema.

        Args:
            text (str): Patient descriptions or conversation fragment.
            context (Union[str, List[Any], dict]): Retrieved context to
              guide the LLM.

        Returns:
            PatientEvidences: Populated Pydantic schema model.
        """
        prompt = PromptTemplate(
            template='''
            You are a medical parser that extracts structured information from patient text.
            
            Important Notes: 
            1. If an evidence is not mentioned in the text, do not include it.
            2. Evidences with data type "B" (Boolean) should have values "YES" or "NO".
            3. Evidences with categorical values should have values as a list of ids. Example: if evidence is "E_55" (eye) and value is "V_125", output ["E_55_@_V_125"].
            4. Similarly, for numerical values, provide the value as a list containing the numerical value string. Example: ["E_59_@_5"].
            5. If multiple values are mentioned for a single evidence, include all relevant ids in the list.
            6. If evidence with data type "M" is present, also include its parent evidence (code_question) with value "YES".
            7. If evidence is explicitly mentioned as absent, set its value to "NO" (e.g., "no fever" -> evidence "E_201" gets value "NO").

            Patient text: "{text}"

            Knowledge Graph Evidences Subset: {context}
            
            {format_instructions}
            ''',
            input_variables=["text", "context"],
            partial_variables={
                "format_instructions": (
                    self.output_parser.get_format_instructions()
                )
            },
        )

        try:
            context_json = (
                context if isinstance(context, str) else json.dumps(context)
            )
            prompt_value = prompt.invoke({
                "text": text,
                "context": context_json,
            })
            
            llm_response = self.llm.invoke(prompt_value)
            
            print("\n" + "=" * 50)
            print("1. BEFORE PYDANTIC (Raw LLM Output)")
            print("=" * 50)
            print(llm_response.content)
            
            parsed_result = self.output_parser.invoke(llm_response)
            
            print("\n" + "=" * 50)
            print("2. AFTER PYDANTIC (Pydantic Object Representation)")
            print("=" * 50)
            print(repr(parsed_result))
            print("\nAs Dictionary:")
            print(json.dumps(parsed_result.model_dump(), indent=2))
            print("=" * 50 + "\n")
            
            return parsed_result
            
        except Exception as e:
            print(f"Error during LLM decoding/parsing: {e}")
            return PatientEvidences(evidences=[], values=[])

    def parse_query(
        self, text: str, context: Union[str, List[Any], dict]
    ) -> Tuple[List[str], List[Any]]:
        """
        Public endpoint implementation of the BaseSymptomParser interface.

        Args:
            text (str): Patient descriptions or conversation fragment.
            context (Union[str, List[Any], dict]): Context details.

        Returns:
            Tuple[List[str], List[Any]]: Parallel lists of evidence keys and
              mapped values.
        """
        parsed_data = self.parser(text, context)
        if parsed_data is None:
            return [], []
        return parsed_data.evidences, parsed_data.values


if __name__ == "__main__":
    parser = Parser()
    G = pickle.load(open("Pickle/kg.pkl", "rb"))
    nlu = DDxGraphNLU(G)
    sample_text = (
        "For the past couple of weeks, I’ve been having sudden episodes "
        "of very intense pain on one side of my head, mainly around my "
        "eye and temple. The pain feels sharp and unbearable, and when "
        "it happens my eye starts watering and my nose feels blocked on "
        "the same side. I can’t stay still during these attacks and feel "
        "extremely restless. These episodes happen multiple times and "
        "often around the same time of day, then completely go away "
        "in between.No fever and cough."
    )
    context = nlu.retrieve(sample_text)
    print("\n" + "=" * 50)
    print("1. RETRIEVED CONTEXT FROM NLU")
    print("=" * 50)
    print(context)
    print("=" * 50 + "\n")
    evidences, values = parser.parse_query(sample_text, context)
    print("\n" + "=" * 50)
    print("3. FINAL EXTRACTED LISTS")
    print("=" * 50)
    print("Evidences:", evidences)
    print("Values:", values)
    print("=" * 50 + "\n")