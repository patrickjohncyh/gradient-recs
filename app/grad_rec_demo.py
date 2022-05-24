import json
import time
import streamlit as st
from fashion_clip.fashion_clip import FCLIPDataset, FashionCLIP
from fashion_clip.utils import display_images_from_url
from gradient_rec import GradREC

@st.cache(allow_output_mutation=True,
          hash_funcs={ GradREC: lambda _ : _.fclip.model_name })
def get_direction_vector(gradrec_model: GradREC, start_query: str, end_query:str):
    return -gradrec_model.direction_vector(start_query, end_query)

@st.cache(allow_output_mutation=True,
          hash_funcs={ GradREC: lambda _ : _.fclip.model_name })
def traverse_space(gradrec_model: GradREC, start_point, search_space, v_dir, step_size, steps, reg_space, reg_weight=0.9):
    @st.cache(allow_output_mutation=True)
    def traverse_space_rec(start_point, v_dir, step_size, steps, reg_weight=0.9):
         if steps == 0:
             return start_point
         return gradrec_model.traversal_fn(start_point=traverse_space_rec(start_point, v_dir, step_size, steps-1, reg_weight,),
                                           v_dir=v_dir,
                                           step_size=step_size,
                                           reg_space=reg_space,
                                           reg_k=100,
                                           reg_weight=reg_weight)

    end_point = traverse_space_rec(start_point, v_dir, step_size, steps, reg_weight)
    nn = gradrec_model._k_neighbors(end_point, search_space, k=100)
    return nn[:10]

def expander_generator(dataset:FCLIPDataset,
                       fclip_model: FashionCLIP,
                       gradrec_model: GradREC,
                       title: str,
                       seed_sku: str,
                       start_query:str = None,
                       end_query:str = None,
                       step_size:float = 2.0,
                       is_example:bool = False,
                       **kwargs):

    text_cols = st.columns([1, 1, 0.75, 0.3])
    im_cols = st.columns([1, 1, 1, 1, 1])
    # Text and Step Input
    with text_cols[0]:
        start_query = st.text_input("Start Query:", value=start_query or '', key=title+'_start_query', disabled=is_example)
    with text_cols[1]:
        end_query = st.text_input("End Query:", value=end_query or '', key=title+'_end_query', disabled=is_example)
    with text_cols[2]:
        step_size = st.number_input("Step Size:",
                                    min_value=0.25,
                                    max_value=10.0,
                                    step=0.25,
                                    value=step_size,
                                    key=title+'_step_size',
                                    disabled=is_example)
    # GO Button
    with text_cols[3]:
        st.markdown('');
        st.markdown('')
        go = st.button('Go!', key=title+'_button')

    # display seed product
    with im_cols[0]:
        st.pyplot(display_images_from_url([dataset._retrieve_row(seed_sku)['product_photo_url']]))

    if not start_query and not end_query:
        return
    if not go:
        return

    # Computation
    v_dir = get_direction_vector(gradrec_model, start_query, end_query)
    seen_prods = [seed_sku]
    for i in range(1,len(im_cols)):
        with im_cols[i]:
            next_product = traverse_space(gradrec_model=gradrec_model,
                                          start_point=fclip_model.image_vectors[dataset.id_to_idx[seed_sku]],
                                          search_space=fclip_model.image_vectors,
                                          v_dir=v_dir,
                                          step_size=step_size,
                                          steps=i,
                                          reg_space=fclip_model.image_vectors,
                                          reg_weight=0.9)
            next_prod_skus = [dataset.ids[idx] for idx in next_product if dataset.ids[idx] not in seen_prods]
            seen_prods.append(next_prod_skus[0])
            product_info = dataset._retrieve_row(next_prod_skus[0])
            print(product_info['category_level_3'])
            st.pyplot(display_images_from_url([product_info['product_photo_url']]))
            time.sleep(0.5)



def app(dataset: FCLIPDataset, fclip_model: FashionCLIP, gradrec_model: GradREC):
    st.write("""
    # GradREC DEMO
    
    In this interactive demo, explore the kinds of comparative recommendations `GradREC` is capable of.
    
    In general, you will need to supply a `start query` and an `end query` whose _semantic difference_ represents
    the attribute/dimension you want to vary. For example, the difference between "Dark Blue Polo T-Shirt" and 
    "Blue Polo T-Shirt" represents the _color luminance_ dimension. In addition, a step size needs to provided, which
    controls the strength of attribute change on each step.
    
    Expand the various sections to view recommendations generated by `GradREC` for various attributes. These come with
    pre-filled values to give you an idea of how `GradREC` works.
     
    In the last section you can select various products and supply your own parameters for generating comparative 
    recommendations.
      
    """)
    with open('data/grad_rec_examples.json') as f:
        examples = json.load(f)

    expanders = [st.expander(eg['title']) for eg in examples if eg['is_example']]
    # examples
    for idx, (exp, ex)in enumerate(zip(expanders, examples)):
        with exp:
          expander_generator(dataset, fclip_model, gradrec_model, **ex)

    # try it out yourself
    diy_expander = st.expander("Try it yourself!")
    with diy_expander:
        prod_sel = st.selectbox('Select a Product', [ _['product_description'] for _ in examples])
        seed_sku =  list(filter(lambda eg: eg['product_description'] == prod_sel, examples))[0]['seed_sku']
        expander_generator(dataset, fclip_model, gradrec_model, title='diy', seed_sku=seed_sku)