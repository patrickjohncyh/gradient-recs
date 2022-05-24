from fashion_clip.fashion_clip import FCLIPDataset, FashionCLIP
from gradient_rec import GradREC


if __name__ == "__main__":
    ff_dataset = FCLIPDataset('FF', image_folder='s3://farfetch-images-ztapq86olwi6kub2p79d/images/')
    fclip = FashionCLIP('FCLIP', ff_dataset)
    grec =  GradREC(fclip)

    query_vectors = grec._encode_queries(['long red skirt', 'short red skirt'])
    v_dir = grec.direction_vector('long red skirt', 'short red skirt')
    start_points, p_info = grec._product_retrieval([query_vectors[0]])
    print(p_info[0][:3])
    path = grec.traverse_space(start_point=start_points[0][0],
                        search_space=fclip.image_vectors,
                        v_dir=v_dir,
                        step_size=0.1,
                        steps=10,
                        reg_space=fclip.image_vectors,
                        reg_weight=0.9)

    for p in path:
        print(fclip.dataset.catalog[p])
        print('---')