import numpy as np
import math
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import annoy  # Use annoy library from Spotify


class EnhancedVectorDatabase():
    """Extends VectorDatabase, so that searching the vector database is done with ANN
        using Spotify's annoy package
    """
    def __init__(self, 
        embedding_model: EmbeddingModel = None, 
        search_in_x_trees=None, 
        distance_measure='cosine'
    ):
        #self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

        # Make sure that the distance measure is supported
        if distance_measure in ["angular", "euclidean", "manhattan", "hamming", "dot", "cosine"]:
            self.distance_measure = distance_measure
        else:
            raise ValueError('Measure distance not supported!')


        self.vectors = []
        self.keys = []        
        self.index = None
        self.search_in_x_trees = search_in_x_trees

        if self.search_in_x_trees is None:
            self.search_in_x_trees = -1
            

    def insert(self, key: str, vector: np.array) -> None:
        """Add a key: vector pair to the class"""
        self.vectors.append(vector)
        self.keys.append(key)


    def build_index(self, number_of_trees=5):
        """Function to build the index before being able to search anything."""
        try:
            dimensions = self.vectors[0].shape[0]
        except Exception as e:
            raise ValueError('Cannot determine the number of dimensions! Did you add any vectors with insert()?')
        
        # If we want cosine distance, we need to get the euclidean distance and then
        # calculate the cosine
        if self.distance_measure == 'cosine':
            distance_measure = 'angular'
        else:
            distance_measure = self.distance_measure

        self.index = annoy.AnnoyIndex(dimensions, distance_measure)

        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)


    def search(
        self,
        query_vector: np.array,
        k: int,
    ) -> List[Tuple[str, float]]:
        """Perform the search"""

        if self.index is None:
            raise RuntimeError('Unable to find an Index! Please run build_index() before searching.')
        
        indices = self.index.get_nns_by_vector(
            query_vector, 
            k, 
            search_k=self.search_in_x_trees,
            include_distances=True,
        )


        if indices is not None:
            result_indices = indices[0]
            result_distances = indices[1]

            if self.distance_measure == 'cosine':
                # We convert the angular distance into cosine similarity with the formula (angular^2)/2 
                # source: https://github.com/spotify/annoy/issues/530
                result_distances = [ ( 1-(math.pow(a,2)/2)) for a in result_distances]

            return [(self.keys[i], d) for i, d in zip(result_indices, result_distances)]
        else:
            return None

    def search_by_text(
        self,
        query_text: str,
        k: int,
        #distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        """Performs search using ANN."""

        # Refuse to search if the index was not built in advance.
        if self.index is None:
            raise RuntimeError('Unable to find an Index! Please run build_index() before searching.')

        # Convert the query to vector using the embedding model
        query_vector = self.embedding_model.get_embedding(query_text)

        # Call the search function
        results = self.search(query_vector, k)

        # Return the top result if return_as_text is positive, otherwise return all top k options
        return [result[0] for result in results] if return_as_text else results


    def retrieve_from_key(self, key: str) -> np.array:
        """Gets the vector corresponding to an exact key."""
        array_keys = np.array(self.keys)
        match_indices = np.where(key == array_keys)[0]

        # Check if any matches were found
        if len(match_indices) > 0:
            # Return the vector at the first matching index
            return self.vectors[match_indices[0]]
        else:
            return None


    async def abuild_from_list(self, list_of_text: List[str]) -> "EnhancedVectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

