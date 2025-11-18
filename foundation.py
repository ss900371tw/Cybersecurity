import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import io
from transformers import cached_path
# 嘗試匯入 pypdf，如果沒有安裝則提示
try:
    import pypdf
except ImportError:
    pypdf = None

# --- 頁面設定 ---
st.set_page_config(page_title="Cybersecurity AI Assistant", page_icon="🛡️", layout="wide")

st.title("🛡️ Foundation-Sec-8B-Instruct Dashboard")
st.markdown("基於 `fdtn-ai/Foundation-Sec-8B-Instruct` 模型的資安專家聊天機器人")

# --- 側邊欄設定 (參數與 Token) ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    default_token = os.getenv("HF_TOKEN", "") 
    hf_token = st.text_input("Hugging Face Token", value=default_token, type="password", help="請輸入您的 HF Token 以存取模型")
    
    st.divider()
    
    # === 新增：檔案上傳功能 ===
    st.subheader("📂 上傳分析檔案")
    uploaded_file = st.file_uploader("上傳 Logs", type=['txt', 'py', 'log', 'csv', 'md', 'json', 'pdf'])
    
    if uploaded_file and uploaded_file.type == "application/pdf" and pypdf is None:
        st.warning("如果要支援 PDF，請安裝 pypdf: `pip install pypdf`")

    st.divider()

    st.subheader("模型參數")
    system_prompt = st.text_area("System Prompt", value="You are a cybersecurity expert. If the user provides a file content, analyze it carefully.", height=100)
    max_new_tokens = st.slider("Max New Tokens", min_value=128, max_value=4096, value=1024, step=128) # 增加上限以容納長檔案分析
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.1, step=0.1, help="數值越低，回答越保守固定；數值越高，回答越有創意。")
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1)
    
    if st.button("清除對話歷史"):
        st.session_state.messages = []
        st.rerun()

# --- 硬體偵測 ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()
st.sidebar.markdown(f"**目前運算裝置:** `{DEVICE}`")

# --- 模型載入 (使用 cache 避免重複載入) ---
@st.cache_resource
def load_model(model_id, token):
    if not token:
        st.error("請在側邊欄輸入 Hugging Face Token")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=token,
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None, None

# 只有在有 Token 時才載入模型
if hf_token:
    MODEL_ID = "fdtn-ai/Foundation-Sec-8B-Instruct"
    with st.spinner(f"正在載入模型 {MODEL_ID} ... (這可能需要幾分鐘)"):
        tokenizer, model = load_model(MODEL_ID, hf_token)
else:
    st.warning("請先輸入 Hugging Face Token 才能開始。")
    st.stop()

# --- 初始化 Session State (對話歷史) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 檔案處理函數 ---
def process_file_content(uploaded_file):
    """讀取上傳檔案並轉為文字字串"""
    if uploaded_file is None:
        return None
    
    file_content = ""
    try:
        # 處理 PDF
        if uploaded_file.type == "application/pdf":
            if pypdf:
                pdf_reader = pypdf.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_content += page.extract_text() + "\n"
            else:
                return "[Error] PDF library not installed."
        # 處理純文字/程式碼/Logs
        else:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_content = stringio.read()
            
        return file_content
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

# --- 顯示對話歷史 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 推論邏輯 ---
def generate_response(prompt, history, sys_prompt, file_context=None):
    # 建構符合 Chat Template 的格式
    messages = [{"role": "system", "content": sys_prompt}]
    
    # 將歷史對話加入
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 如果有檔案內容，將其組合進 Prompt 中
    full_user_input = prompt
    if file_context:
        full_user_input = f"""I have uploaded a file. Here is the content:
        
=== BEGIN FILE CONTENT ===
{file_context}
=== END FILE CONTENT ===

User Question: {prompt}
"""
    
    # 加入當前使用者輸入
    messages.append({"role": "user", "content": full_user_input})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 注意：如果檔案太長，這裡可能會超過模型上限，實際生產環境需要做截斷處理
    inputs_tokenized = tokenizer(inputs, return_tensors="pt")
    input_ids = inputs_tokenized["input_ids"].to(DEVICE)
    
    do_sample = True
    current_temp = temperature
    if temperature == 0:
        do_sample = False
        current_temp = None 

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": current_temp,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            **generation_args,
        )
    
    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], 
        skip_special_tokens=True 
    )
    
    return response

# --- 處理使用者輸入 ---
if prompt := st.chat_input("請輸入關於資安的問題..."):
    
    # 1. 處理檔案
    file_text = None
    display_prompt = prompt # 在畫面上顯示的文字
    
    if uploaded_file:
        with st.spinner("正在讀取檔案內容..."):
            file_text = process_file_content(uploaded_file)
            if file_text:
                # 如果有檔案，我們在畫面上加個小提示，但不要把整個檔案內容印出來洗版
                display_prompt = f"📄 **[已附加檔案: {uploaded_file.name}]**\n\n{prompt}"
                # 簡單的長度檢查警告
                if len(file_text) > 20000:
                    st.toast("⚠️ 檔案內容較長，可能會超過模型處理上限。", icon="⚠️")

    # 2. 顯示使用者訊息
    st.chat_message("user").markdown(display_prompt)
    
    # 3. 呼叫模型產生回應
    if model and tokenizer:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("正在分析與思考中..."):
                # 傳入 file_text 作為額外上下文
                response = generate_response(prompt, st.session_state.messages, system_prompt, file_context=file_text)
                message_placeholder.markdown(response)
        
        # 4. 更新對話歷史
        # 這裡我們選擇儲存 display_prompt，讓歷史紀錄看得到有傳檔案，但模型實際上是收到完整文字
        # 注意：為了節省 Context，歷史紀錄裡我們不存完整的檔案內容，只存使用者的問題
        # 如果希望模型在"下一輪"對話還記得檔案，則必須將 full content 存入 history，但這會消耗大量記憶體
        st.session_state.messages.append({"role": "user", "content": display_prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
