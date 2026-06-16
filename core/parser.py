import json
import os
import pickle
import time
from typing import Any, List, Tuple, Union

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from tenacity import Retrying, stop_after_attempt, wait_fixed

from core.config import load_config
from core.interfaces import BaseSymptomParser
from core.nlu import DDxGraphNLU

config = load_config()
DEFAULT_MODEL_NAME = config["parser"]["model_name"]
FALLBACK_MODELS = config["parser"]["fallback_models"]
MAX_RETRIES = config["parser"]["max_retries"]
RETRY_DELAY = config["parser"]["retry_delay"]

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

    def __init__(
        self,
        model_name: str = None,
        fallback_models: List[str] = None,
        max_retries: int = None,
        retry_delay: float = None,
    ):
        """
        Initialize the LLM parser.

        Args:
            model_name (str, optional): LLM model identifier.
            fallback_models (list, optional): Alternative fallback models.
            max_retries (int, optional): Max retry attempts per model.
            retry_delay (float, optional): Delay in seconds between retries.
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.fallback_models = (
            fallback_models
            if fallback_models is not None
            else FALLBACK_MODELS
        )
        self.max_retries = (
            max_retries if max_retries is not None else MAX_RETRIES
        )
        self.retry_delay = (
            retry_delay if retry_delay is not None else RETRY_DELAY
        )
        self.llm = self._create_llm(self.model_name)
        self.output_parser = PydanticOutputParser(
            pydantic_object=PatientEvidences
        )

    def _create_llm(self, model_name: str) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance with the specified model name.
        """
        timeout = config["parser"]["request_timeout"]
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
            base_url="https://router.huggingface.co/v1",
            temperature=0.0,
            timeout=timeout
        )

    def parser(
        self, text: str, context: Union[str, List[Any], dict]
    ) -> PatientEvidences:
        """
        Invoke LLM to extract structured patient symptoms schema.
        Includes automatic retry and fallback logic.

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

        context_json = (
            context if isinstance(context, str) else json.dumps(context)
        )
        prompt_value = prompt.invoke({
            "text": text,
            "context": context_json,
        })

        models_to_try = [self.model_name] + self.fallback_models

        for model_idx, model in enumerate(models_to_try):
            llm_instance = (
                self.llm
                if model == self.model_name
                else self._create_llm(model)
            )

            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_fixed(self.retry_delay),
                    reraise=True,
                ):
                    with attempt:
                        # Print current model attempt count
                        attempt_num = attempt.retry_state.attempt_number
                        print(
                            f"[Parser] Querying model '{model}' "
                            f"(Attempt {attempt_num}/{self.max_retries})..."
                        )
                        llm_response = llm_instance.invoke(prompt_value)

                        print("\n" + "=" * 50)
                        print(
                            f"1. BEFORE PYDANTIC (Raw LLM Output - "
                            f"Model: {model})"
                        )
                        print("=" * 50)
                        print(llm_response.content)

                        parsed_result = self.output_parser.invoke(
                            llm_response
                        )

                        print("\n" + "=" * 50)
                        print(
                            "2. AFTER PYDANTIC (Pydantic Object "
                            "Representation)"
                        )
                        print("=" * 50)
                        print(repr(parsed_result))
                        print("\nAs Dictionary:")
                        print(
                            json.dumps(
                                parsed_result.model_dump(), indent=2
                            )
                        )
                        print("=" * 50 + "\n")

                        return parsed_result

            except Exception as e:
                print(
                    f"[Parser] Model '{model}' failed completely "
                    f"after {self.max_retries} attempts: {e}"
                )
                if model_idx < len(models_to_try) - 1:
                    next_model = models_to_try[model_idx + 1]
                    print(
                        f"[Parser] Fallback: Switching to next model "
                        f"'{next_model}'."
                    )

        print("[Parser] Critical: All LLM models and retries failed.")
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