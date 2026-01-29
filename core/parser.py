import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
    release_evidences = json.load(f)

class Parser:
    def __init__(self, model_name="gpt-oss:120b"):
        self.model_name = model_name

    def parser(self, text):
        prompt = f'''
        You are a parser that extracts structured information from text.
        Important Notes: 
        1. If an evidence is not mentioned in the text, do not include it in the output.
        2. Evidences with data type "B"(Boolean) should have values "YES" or "NO".
        3. Evidences with categorical values should have values as a list of ids. For example, if the evidence is "E_55" and the mentioned value is "eye", and in release_evidences.json "eye" maps to "V_125", then the value should be ["E_55_@_V_125"].
        4. Similarly for evidences with numerical values, provide the value as a list containing the numerical value as a string. For example, if the evidence is "E_59" and the value maps to a numerical value "5" (1-10), then the value should be ["E_59_@_5"].
        5. If multiple values are mentioned for a single evidence, include all relevant ids in the list.
        6. If evidence with data type "M" is present, then also include its parent evidence(code_question) with value "YES".
        7. If evidence is mentioned as absent, set its value to "NO". For example, if the text says "no fever", then for evidence "E_201" (fever), set value to "NO".
        8. Ensure the output is a valid JSON.
        9. The output JSON should have two keys: "evidences" and "values". "evidences" is a list of evidence ids, and "values" is a list of corresponding values.

        For the below patient text, map the evidences and values ids from release_evidences.json file. 

        Patient text: "{text}"

        release_evidences.json: {release_evidences}
        '''

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN")
        )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        print(completion.choices[0].message.content)

        try:       
            content = completion.choices[0].message.content
            if content is None:
                print("Error: API returned empty content")
                return None
            clean = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            parsed_data = json.loads(clean)
        except Exception as e:
            print(f"Error during decoding: {e}")
            return None


        return parsed_data
         
    
    def parse_query(self, text):
        parsed_data = self.parser(text)
        if parsed_data is None:
            return [], []

        evidences = parsed_data.get("evidences", [])
        values = parsed_data.get("values", [])

        return evidences, values
  

if __name__ == "__main__":
    parser = Parser()
    sample_text = "For the past couple of weeks, I’ve been having sudden episodes of very intense pain on one side of my head, mainly around my eye and temple. The pain feels sharp and unbearable, and when it happens my eye starts watering and my nose feels blocked on the same side. I can’t stay still during these attacks and feel extremely restless. These episodes happen multiple times and often around the same time of day, then completely go away in between.No fever and cough."
    evidences, values = parser.parse_query(sample_text)
    print("Evidences:", evidences)
    print("Values:", values)