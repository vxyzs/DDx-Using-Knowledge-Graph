import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr


@st.cache_data(show_spinner=False)
def refine_question(original_question):
    """
    Refines a raw medical question so that it is simple and clear for patients.
    """
    llm = ChatOpenAI(
        model="openai/gpt-oss-safeguard-20b",
        api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
        base_url="https://router.huggingface.co/v1",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template='''You are a helpful medical assistant. Refine the following medical question so that a patient can easily understand it. Keep it as short and direct as possible. Do not change the medical meaning. Only output the question text itself.

Original Question: {question}
Refined Question:''',
        input_variables=["question"]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"question": original_question})
        return response.content.strip(" \n\"'")
    except Exception as e:
        return original_question


def generate_explanation(results):
    """
    Generates a medically explainable differential diagnosis report.
    """
    llm = ChatOpenAI(
        model="openai/gpt-oss-safeguard-20b",
        api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
        base_url="https://router.huggingface.co/v1",
        temperature=0.7
    )
    
    prompt = PromptTemplate(
        template='''You are a medical diagnostic assistant.

Based on the provided diagnostic results (which include top conditions, supporting symptoms, and absent symptoms), generate a brief, professional, and medically explainable differential diagnosis summary.

Return the output as a markdown table with the following columns EXACTLY:

| Condition | Probability | Why It Is Suspected | Missing / Contradicting Evidence | Recommended Next Steps |

Instructions:
- Use the condition name and probability exactly as provided.
- In **Why It Is Suspected**, analyze the supporting evidence (the confirmed symptoms listed for this condition) and briefly explain the clinical reasons and symptom patterns.
- In **Missing / Contradicting Evidence**, analyze the missing evidence (the absent symptoms listed for this condition) and mention important absent or conflicting findings that reduce confidence.
- In **Recommended Next Steps**, suggest practical next actions such as monitoring symptoms, additional questions, medical tests, referral, or urgent care if red flags exist.
- Keep explanations concise, medically accurate, evidence-based, and easy for a patient to understand.
- Do NOT present any condition as confirmed.
- Do NOT invent symptoms or evidence not provided.

After the table, include a short 2–3 line summary explaining:
1. the most likely condition,
2. remaining uncertainty,
3. what additional evidence would improve confidence.

End with:
**“This assessment is assistive and does not replace evaluation by a qualified healthcare professional.”**

Diagnostic Results:
{results}

Explanation:''',
        input_variables=["results"]
    )
    
    chain = prompt | llm
    try:
        response = chain.invoke({"results": str(results)})
        return response.content
    except Exception as e:
        return f"Could not generate explanation due to an error: {e}"


def answer_clarification_question(system_question, evidence_id, user_query, release_evidences):
    """
    Explains the current diagnostic question to the patient in patient-friendly language.
    """
    llm = ChatOpenAI(
        model="openai/gpt-oss-safeguard-20b",
        api_key=SecretStr(os.getenv("HF_TOKEN") or ""),
        base_url="https://router.huggingface.co/v1",
        temperature=0.3
    )
    
    # Extract metadata descriptions if available
    symptom_details = ""
    if evidence_id and evidence_id in release_evidences:
        meta = release_evidences[evidence_id]
        symptom_details = f"Symptom Code: {evidence_id}\nSymptom Name (EN): {meta.get('question_en', '')}"
        value_meanings = meta.get("value_meaning", {})
        if value_meanings:
            meanings_str = ", ".join(f"{k}: {v.get('en', '')}" for k, v in value_meanings.items())
            symptom_details += f"\nPossible values: {meanings_str}"

    prompt = PromptTemplate(
        template='''You are a medical assistant helping a patient understand a diagnostic question asked by their AI doctor.

Context:
Doctor's Question: "{system_question}"
{symptom_details}

Patient's Clarification Query: "{user_query}"

Instructions:
- Explain what the symptom or question means in simple, patient-friendly language.
- Give concrete, everyday examples if applicable.
- Briefly tell the patient how they can verify or check if they have this symptom (e.g., feeling their pulse, looking in the mirror, checking for tenderness).
- Keep your response brief, clear (2-4 sentences max), comforting, and medically accurate. Do not suggest a diagnosis.

Explanation:''',
        input_variables=["system_question", "symptom_details", "user_query"]
    )
    chain = prompt | llm
    try:
        response = chain.invoke({
            "system_question": system_question,
            "symptom_details": symptom_details,
            "user_query": user_query
        })
        return response.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't process your request: {e}"
