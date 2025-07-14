import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
from utils import preprocess_image, segment_lines

@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    return processor, model

st.set_page_config(page_title="Handwritten OCR", layout="centered")
st.title("✍️ Handwritten Text to OCR using Hugging Face (TrOCR)")
st.write("Upload a handwritten image and extract text using Transformer-based OCR")

uploaded_file = st.file_uploader("📤 Upload a handwritten image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🖼 Uploaded Image")
    st.image(image, caption="Handwritten Input", use_container_width=True)

    processor, model = load_model()

    st.subheader("🔍 Line Segmentation")
    lines = segment_lines(image)

    if not lines:
        st.warning("No text lines detected.")
    else:
        st.success(f"{len(lines)} line(s) detected")
        st.subheader("📝 Predicted Text")

        final_output = ""
        for idx, line_img in enumerate(lines):
            preprocessed = preprocess_image(line_img).convert("RGB")
            pixel_values = processor(images=preprocessed, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated_ids = model.generate(pixel_values)

            raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            cleaned_text = raw_text.strip()
            cleaned_text = re.sub(r"[\d\W_]+$", "", cleaned_text)

            st.markdown(f"**Line {idx + 1}:** {cleaned_text}")
            final_output += cleaned_text + "\n"
