from typing import List, Union
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from tqdm import tqdm

_PROMPT_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'a dark photo of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
]

class GradREC:

    def __init__(self, fclip: FashionCLIP):
        self.fclip = fclip

    def _encode_queries(self, queries: List[str]):
        """
        Encode queries into vector representations whilst applying prompt templates

        :param queries: list of queries to encode
        :return: textual vector representation for queries
        """
        # TODO: offload prompt template computation to FashionCLIP
        # apply prompt templates to query
        queries_with_prompt = [[prompt.format(q) for prompt in _PROMPT_TEMPLATES] for q in queries]
        # flatten queries with prompt and encode in batch
        text_vectors_flat = self.fclip.encode_text([ _ for q in queries_with_prompt for _ in q], batch_size=32)
        # un-flatten vectors to correct shape
        text_vectors = text_vectors_flat.reshape(len(queries), len(_PROMPT_TEMPLATES), -1)
        # compute average for each query
        query_vectors = text_vectors.mean(axis=1)
        return query_vectors

    def _product_retrieval(self, query_vectors: np.ndarray, N:int =100):
        """
        Retrieve product image vectors for query vectors using cosine similarity as the distance measure

        :param query_vectors: vectors used as retrieval keys
        :param N: number of products to retrieve
        :return: image vector representation
        """
        cosine_sim = self.fclip._cosine_similarity(query_vectors, self.fclip.image_vectors, normalize=True)
        indices = cosine_sim.argsort()[:, -N:][:, ::-1]
        return self.fclip.image_vectors[indices]

    def direction_vector(self, start_query: str, end_query: str, start_N=100, end_N=1000):
        """
        Computes direction vector given start and end query

        :param start_query: start query text
        :param end_query: end query text
        :param start_N: number of products to retrieve for start query
        :param end_N: number of products to retrieve for end query
        :return: direction vector for traversal
        """
        # TODO: Simplify computation by leveraging FashionCLIP methods
        # TODO: Generalize to any latent space
        # encode start and end queries
        query_vectors = self._encode_queries([start_query, end_query])
        # retrieve nearest image vectors
        query_im_vectors = self._product_retrieval(query_vectors, N=max(start_N, end_N))
        return self._direction_vector_from_vectors(query_im_vectors[0][:start_N], query_im_vectors[1][:end_N])

    def _direction_vector_from_vectors(self,
                                       exemplar_vectors: np.ndarray,
                                       pop_vectors: np.ndarray,
                                       normalize:bool = True):
        """
        Computes the channel-wise SNR given exemplar and population vectors

        :param exemplar_vectors: exemplar vectors
        :param pop_vectors: population vectors
        :param normalize: flag to normalize SNR vector
        :return: channel-wise SNR vector
        """

        population_mean = np.mean(pop_vectors, axis=0, keepdims=True)
        population_std = np.std(pop_vectors, axis=0, keepdims=True)

        deltas = (exemplar_vectors - population_mean) / population_std
        delta_mean = np.mean(deltas, axis=0, keepdims=False)
        delta_std = np.std(deltas, axis=0, keepdims=False)
        theta = delta_mean / delta_std
        v_dir = theta / np.linalg.norm(theta, ord=2, axis=-1) if normalize else theta

        return v_dir

    def _k_neighbors(self,
                     point: np.ndarray,
                     space, k=10):
        """
        Compute nearest neighbor given point in latent space

        :param point: seed point in space
        :param space: latent space to perform NN retrieval on
        :param k: number of neighbors to retrieve
        :return: k neighbors of a given point
        """
        # TODO: Simplify or remove method by using FashonCLIP methods directly
        cosine_sim = self.fclip._nearest_neighbours(k, [point], space)[0]
        return cosine_sim

    def traversal_fn(self,
                     start_point: np.ndarray,
                     v_dir: np.ndarray,
                     step_size: float,
                     reg_space: np.ndarray,
                     reg_weight: float,
                     reg_k: int,
                     nearest_neighbors=None):
        """
        Single-step latent space traversal

        :param start_point: starting point for traversal
        :param v_dir: direction vector used for traversal
        :param step_size: size of step
        :param reg_space: latent space used for regularization
        :param reg_weight: regularization weight
        :param reg_k: number of neighbors used for regularization
        :param nearest_neighbors: option to pass in pre-computed nearest neighbors for regularization
        :return: new point in space after traversal
        """
        start_point = start_point / np.linalg.norm(start_point, ord=2)
        if nearest_neighbors is None:
            nearest_neighbors = self._k_neighbors(start_point, reg_space, k=reg_k)
        neighborhood_mean  = reg_space[nearest_neighbors].mean(axis=0)
        return start_point + (1 - reg_weight) * step_size * v_dir + reg_weight * neighborhood_mean

    def traverse_space(self,
                       start_point: np.ndarray,
                       search_space: np.ndarray,
                       v_dir: np.ndarray,
                       step_size: float,
                       steps: int,
                       reg_space: np.ndarray,
                       reg_weight: float,
                       reg_k: int = 100,
                       k=10):
        """

        :param start_point: starting point for traversal
        :param search_space: latent space to traverse
        :param v_dir: direction vector used for traversal
        :param step_size: size of step
        :param steps: number of traversal steps
        :param reg_space: latent space used for regularization
        :param reg_weight: regularization weight
        :param reg_k: number of neighbors used for regularization
        :param k: number of products to return for each traversal step
        :return: list of list of product indices for each step in traversal
        """
        nearest_neighbors = self._k_neighbors(start_point, search_space, k=reg_k)
        traversal_path = [nearest_neighbors[:k]]
        for _ in tqdm(range(steps)):
            # take a single step
            start_point = self.traversal_fn(
                start_point=start_point,
                v_dir=v_dir,
                step_size=step_size,
                reg_space=reg_space,
                reg_weight=reg_weight,
                reg_k=reg_k,
                nearest_neighbors=nearest_neighbors)
            # get nearest products
            nearest_neighbors = self._k_neighbors(start_point, search_space, k=reg_k)
            # store products
            traversal_path.append(nearest_neighbors[:k])
        return traversal_path



