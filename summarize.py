from langchain_community.llms import OpenAI
import os
import PIL
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import easyocr
from langchain.prompts import PromptTemplate
st.header("Summarizer")
reader = easyocr.Reader(['en'])
uploaded_file=st.file_uploader("Upload image:")
llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.7)
pt=PromptTemplate(input_variables=["text"],template="Read the following text: {text} and give the important points in the form of bullet points. ")
from langchain.chains import LLMChain
chain=LLMChain(llm=llm,prompt=pt)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    result = reader.readtext(np.array(image), detail = 0)
    st.subheader("OCR result:")
    st.write(result)
    st.subheader("Summary: ")
    st.write(chain.invoke(result)["text"])