import streamlit as st
import pandas as pd
from utils import plotly_tsne, resize_gif
from fashion_clip.fashion_clip import FCLIPDataset
from fashion_clip.utils import display_images_from_s3
import base64



def app():
    st.write("""
    # Introduction

    Item-to-item comparative recommendations of the form "Can I get something _darker_/_longer_/_warmer_?" traditionally
    require fine-grained supervised data and employ "learning-to-rank" approaches. In this demo we introduce a zero-shot
    approach toward generating comparative recommendations by leveraging the linguistic capabilites of `FashionCLIP`, a
    `CLIP`-like model fine-tuned for fashion concepts.
    
    
    #### `FashionCLIP` latent space
    
    We begin by first exploring the latent space of `FashionCLIP`. By selecting different options in the drop down below
    you can visualize the TSNE projects of the image embeddings for various _comparative concepts_.    
    """)

    # Latent Space Visualization
    tsne_plots = {
        'Shirt Color Luminance' : pd.read_csv('data/tsne_fclip_blue_shirt.csv'),
        'Skirt Length': pd.read_csv('data/tsne_fclip_skirt_length.csv'),
        'Footwear Formality': pd.read_csv('data/tsne_fclip_shoes.csv')
    }
    tsne_sel = st.selectbox(label='', options=list(tsne_plots.keys()))
    st.plotly_chart(plotly_tsne(tsne_plots[tsne_sel],
                                title=tsne_sel,
                                enable_legend=True))

    st.write("""
        We observe that `FashionCLIP` organizes the various intensities for each comparative concept into separate clusters, 
        suggesting that it is possible to trace a path in the latent space from one cluster to another. For example for the
        comparative concept of ___shirt color luminance___, three distinct clusters are formed for 
        __light blue polo t-shirt__, __blue polo t-shirt__, and __dark blue polo t-shirt__. Similar patterns are observed
        for the other examples.
        
        The _goal_ of `GradREC` is to traverse such a path, in order to discover products along a certain comparative dimension. 
    """)

    # Method

    st.write("""
    #### Method
    
    We visualize here the overarching approach toward traversing the latent space. 
    
    The two main ingredients are:
    
    1. __Traversal Function__: Takes in an existing point in space (1) and a traversal vector (2), and returns a new point in space 
       which is closer to a product (3) with increasing/decreasing attribute intensity (e.g. decreasing skirt length). Repeated application
       allows for traversal of the space. See illustration below.
   """)
    gif_path = "data/GradREC_gif.gif"
    gif_mini_path = "data/GradREC_gif_mini.gif"
    # resize_gif(gif_path, gif_mini_path, scale=3.5)
    with open(gif_mini_path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    exp_func = st.expander("Traversal Function")
    exp_func.markdown(
        """
        <img src="data:image/gif;base64,{}" alt="cat gif" witdth=100 height=auto>
        &nbsp  
        &nbsp  
        """.format(data_url),
        unsafe_allow_html=True)

    st.write("""
    2. __Traversal Vector__: We construct the traversal vector by borrowing ideas from literature. The intuition behind 
        the approach is to find channels which encode the attribute of interest.
    """)

    gif_path = "data/GradREC_traversal_vector_gif.gif"
    gif_mini_path = "data/GradREC_traversal_vector_gif_mini.gif"
    # resize_gif(gif_path, gif_mini_path, scale=5.5)
    with open(gif_mini_path, "rb") as f:
        contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    exp_vector = st.expander("Traversal Vector")
    exp_vector.markdown(
        """
        <img src="data:image/gif;base64,{}" alt="cat gif" witdth=100 height=auto>  
        """.format(data_url),
        unsafe_allow_html=True)




