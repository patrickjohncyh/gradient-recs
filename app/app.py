import streamlit as st
import grad_rec_demo
import intro

from fashion_clip.fashion_clip import FCLIPDataset, FashionCLIP
from gradient_rec import GradREC
from dotenv import load_dotenv

load_dotenv("../.env")

PAGES = {
    "Intro": intro,
    "Demo": grad_rec_demo,
}

@st.cache(allow_output_mutation=True)
def load_model():
    dataset = FCLIPDataset('FF',
                           image_source_path='s3://farfetch-images-ztapq86olwi6kub2p79d/images/',
                           image_source_type='S3')
    fclip_model = FashionCLIP('FCLIP', dataset)
    grad_rec_model = GradREC(fclip_model)
    print('DONE LOADING MODEL')
    return dataset, fclip_model, grad_rec_model

page = st.sidebar.selectbox("", list(PAGES.keys()))
if page == "Intro":
    PAGES[page].app()
else:
    DATASET, FCLIP, GRADREC = load_model()
    PAGES[page].app(DATASET, FCLIP, GRADREC)
