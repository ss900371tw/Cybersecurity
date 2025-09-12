# app.py
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import io
from huggingface_hub import login
import os

foundation_token = os.getenv("foundation_token","")
login(token=foundation_token)

# ✅ 初始化模型 (只要跑一次)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("fdtn-ai/Foundation-Sec-8B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.to(device), device

tokenizer, model, device = load_model()

def analyze_alert(alert_dict):
    """將單筆 log dict 丟進模型產生 MITRE ATT&CK Mapping"""
    prompt = f"""
You are a cybersecurity analyst. Based on the following network alert, map it to MITRE ATT&CK techniques.

Alert: {alert_dict}

Please **list only the technique name and ID**, one per line.
Example output format:
Command and Scripting Interpreter: Web Shell (T1505.003)
Exploitation of Web Application (T1190)
Remote Services (T1021)
"""
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=300,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response


# ✅ Streamlit UI
st.title("🔎 Cybersecurity Log MITRE ATT&CK Mapper")

uploaded_files = st.file_uploader(
    "上傳 Log 檔案 (csv / txt / log)", type=["csv", "txt", "log"], accept_multiple_files=True
)

if uploaded_files:
    st.success(f"已上傳 {len(uploaded_files)} 個檔案")

if st.button("🚀 開始分析"):
    results = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8", errors="ignore")

        # 嘗試讀取成 CSV，否則逐行讀
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            lines = content.splitlines()
            df = pd.DataFrame({"raw_log": lines})

        st.subheader(f"📄 分析檔案: {uploaded_file.name}")

        # 假設每一行是一筆 log，你可以依需求調整 parsing
        for idx, row in df.iterrows():
            alert_dict = {}
            if "raw_log" in row:
                alert_dict["payload"] = row["raw_log"]
            else:
                alert_dict = row.to_dict()

            response = analyze_alert(alert_dict)
            results.append({"file": uploaded_file.name, "log_index": idx, "mitre_mapping": response})

        st.dataframe(pd.DataFrame(results))
            response = analyze_alert(alert_dict)
            results.append({"log_index": idx, "mitre_mapping": response})

        st.dataframe(pd.DataFrame(results))
