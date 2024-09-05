from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from datetime import datetime
import faiss
import numpy as np
import requests


class BitextBenchmark:
    def __init__(
        self,
        dataset_name: str = "default",
        base_url: str = "http://localhost:8000/v2/models",
        embedding_dim: int = 1024,
        faiss_factory_str: str = "Flat",
        max_workers: int = 60,
    ) -> None:
        self.dataset_name = dataset_name
        self.base_url = base_url
        self.embedding_dim = embedding_dim
        self.max_workers = max_workers

        self._load_dataset()
        self._create_embeddings()
        self._build_indices(faiss_factory_str)
        # Change for default of 16 to improve results
        if faiss_factory_str == "HNSW" or faiss_factory_str == "HNSW32":
            self.index_1.hnsw.efSearch = 32
            self.index_2.hnsw.efSearch = 32

    def _load_dataset(self) -> None:
        dataset = load_dataset(
            "mteb/bucc-bitext-mining", self.dataset_name, split="test"
        )

        self.sentences_1 = [f"query: {s}" for s in dataset["sentence1"]]
        self.sentences_2 = [f"query: {s}" for s in dataset["sentence2"]]

    def _create_embeddings(self) -> None:
        print("Creating embeddings...")
        start = datetime.now()
        self.embeddings_1 = self._parallel_embed(self.sentences_1)
        self.embeddings_2 = self._parallel_embed(self.sentences_2)
        end = datetime.now()
        print(f"Embedding creation took {end-start}")

    def _build_indices(self, faiss_factory_str: str) -> None:
        print("Building indices...")
        start = datetime.now()
        self.index_1 = self._create_faiss_index(self.embeddings_1, faiss_factory_str)
        self.index_2 = self._create_faiss_index(self.embeddings_2, faiss_factory_str)
        end = datetime.now()
        print(f"Index building took {end-start}")

    def _create_faiss_index(self, embeddings: np.ndarray, faiss_factory_str: str):
        index = faiss.index_factory(
            self.embedding_dim, faiss_factory_str, faiss.METRIC_INNER_PRODUCT
        )
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
        return index

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Submit single string of text to be embedded by the multilingual_e5_large
        endpoint.

        Parameters
        ----------
        text : str
            Text to be embedded

        Returns
        -------
        embedding: np.ndarray
            Embedding vector. Shape=[d], dtype=np.float32
        """
        inference_json = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        response = requests.post(
            url=f"{self.base_url}/embed_text/infer", json=inference_json
        )

        return np.array(response.json()["outputs"][0]["data"]).astype(np.float32)

    def _parallel_embed(self, sentences: list[str]) -> np.ndarray:
        """
        Submit sentences to be embedded to the multilingual_e5_large deployment
        endpoint using a threadpool.

        Parameters
        ----------
        sentences : list[str]
            List of sentences to be embedded.

        Returns
        -------
        embeddings: np.ndarray
            Embedding vectors. Shape=[len(sentences), d], dtype=np.float32
        """
        n_sentences = len(sentences)
        embeddings = np.zeros((n_sentences, self.embedding_dim), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, embedding in enumerate(executor.map(self.get_embedding, sentences)):
                embeddings[i] = embedding
        return embeddings

    def margin_score(self, e1: np.ndarray, e2: np.ndarray, k: int = 4) -> float:
        """
        Calculates a margin-based score taken from https://arxiv.org/pdf/1811.01136
        (see Section 3.1) between the cosine of a given candidate and the average
        cosine of its k nearest neighbors in both directions.

        This score is used to reorder a list of candidate neighbors of e1 when
        calculating the top-1 accuracy.

        This appears to be the approach used for the BUCC Bitext Mining dataset.

        Parameters
        ----------
        e1 : np.ndarray
            First embedding vector
        e2 : np.ndarray
            Second embedding vector
        k : int, optional
            Number of nearest neighbors to consider, by default 4

        Returns
        -------
        float
            Ratio of the cosine(e1, e2) over the sum of the ave neighbors
        """
        cosine = e1.dot(e2)
        ave_e1_neighbors = self._get_average_neighbor_score(e1, self.index_2, k)
        ave_e2_neighbors = self._get_average_neighbor_score(e2, self.index_1, k)

        return cosine / (ave_e1_neighbors + ave_e2_neighbors)

    def _get_average_neighbor_score(
        self, e: np.ndarray, index: faiss.IndexFlatIP, k: int
    ) -> float:
        D, _ = index.search(e.reshape(1, -1), k)
        return D.sum() / (2 * k)

    def top_1_accuracy(self, k_candidates: int = 4):
        """
        Get top-1 accuracy using both the margin scoring and straight cosine

        Parameters
        ----------
        k_candidates : int, optional
            Size of the candidate pool to consider, by default 4

        Returns
        -------
        float, float
            margin accuracy, cosine accuracy
        """
        n_correct = 0
        n_correct_cosine = 0
        total = len(self.embeddings_1)

        _, I_batch = self.index_2.search(self.embeddings_1, k_candidates)

        for idx_1, (I, e1) in enumerate(zip(I_batch, self.embeddings_1)):
            if idx_1 == I[0]:
                n_correct_cosine += 1

            scores = [
                self.margin_score(e1, self.embeddings_2[idx_2], k_candidates)
                for idx_2 in I
            ]
            nearest_neighbor = I[np.argmax(scores)]

            if idx_1 == nearest_neighbor:
                n_correct += 1

            if (idx_1 + 1) % 3000 == 0:
                print(
                    f"Processed {idx_1+1}/{total}, Accuracy: {n_correct / (idx_1 + 1):.4f}"
                )

        return n_correct / total, n_correct_cosine / total


def main():
    totals = []
    accuracy_margins = []
    accuracy_cosines = []
    for lang_pair in ["zh-en", "fr-en", "de-en", "ru-en"]:
        print(f"Starting on {lang_pair}")
        benchmark = BitextBenchmark(lang_pair, faiss_factory_str="HNSW32")
        accuracy_margin, accuracy_cosine = benchmark.top_1_accuracy()
        total = benchmark.embeddings_1.shape[0]
        print(
            f"\n{lang_pair} {total=:,} {accuracy_margin=:.4f} {accuracy_cosine=:.4f}\n"
        )
        totals.append(total)
        accuracy_margins.append(accuracy_margin)
        accuracy_cosines.append(accuracy_cosine)

    totals = np.array(totals)
    accuracy_margins = np.array(accuracy_margins)
    accuracy_cosines = np.array(accuracy_cosines)

    mean_accuracy_margins = (totals * accuracy_margins).sum() / totals.sum()
    mean_accuracy_cosines = (totals * accuracy_cosines).sum() / totals.sum()

    print(f"{mean_accuracy_margins=:.4}, {mean_accuracy_cosines=:.4}")


if __name__ == "__main__":
    main()
