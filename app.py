import streamlit as st
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import re, string
import numpy as np

# Text preprocessing
def filtering_text(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/\S+','',text)
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r'(b\'{1,2})', "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model and tokenizer from Hugging Face Hub
@st.cache_resource
def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("raihannnRama/IndoBERT-AnxietyDepression")
    model = BertForSequenceClassification.from_pretrained("raihannnRama/IndoBERT-AnxietyDepression")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = get_model_and_tokenizer()

# Streamlit UI
st.title("Klasifikasi Mental Health Depression dan Anxiety")
user_input = st.text_area("Masukan teks yang ingin dianalisis",)
button = st.button("Prediksi")

label_map = {
    0: "Depression",
    1: "Anxiety"
}

if button and user_input:
    clean_text = filtering_text(user_input)
    inputs = tokenizer(
        [clean_text],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (probs >= 0.5).astype(int)

    # Determine prediction label
    if (pred == [1, 0]).all():
        label_text = "Hasil prediksi model:  <b>Depression</b>"
    elif (pred == [0, 1]).all():
        label_text = "Hasil prediksi model:  <b>Anxiety</b>"
    elif (pred == [1, 1]).all():
        label_text = "Hasil prediksi model:  <b>Depression dan Anxiety</b>"
    else:
        label_text = "Hasil prediksi model:  <b>Tidak keduanya</b>"

    # Styled centered output
    st.markdown(
        f"""
        <div style='
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #2c2c2c;
            color: white;
            padding: 1.5em;
            border-radius: 10px;
            margin-top: 2em;
            font-size: 1.5em;
            text-align: center;
        '>
            {label_text}
        </div>
        """,
        unsafe_allow_html=True
    )
