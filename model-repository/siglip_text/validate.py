from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
from pathlib import Path
import requests
from sklearn import neighbors, metrics


def get_labels(synset_mapping: Path):
    """
    Get the synset category labels and their corresponding text description from
    the LOC_synset_mapping.txt file

    Parameters
    ----------
    synset_mapping : Path
        LOC_synset_mapping.txt file location

    Returns
    -------
    labels: list
        List of the synset category labels (e.g, n01742172)
    labels_text: dict
        Synset category label (e.g, n01742172) as the keys and the text description
        (eg., 'boa constrictor, Constrictor constrictor') as the values.
    """
    labels = []
    labels_text = {}
    with synset_mapping.open("r") as f:
        for line in f:
            line_parts = line.split()
            label = line_parts[0]
            text = " ".join(line_parts[1:])
            labels.append(label)
            labels_text[label] = text
    labels = np.array(labels)
    return labels, labels_text


def get_image_embeddings(
    input_dir: Path, n_categories: int = 0, n_samples_per_category: int = 0
):
    """
    Iterate through input directory of images and submit them to Triton Inference
    Server to get their image embeddings. The category label is the first part of the
    file's name.

    Parameters
    ----------
    input_dir : Path
        Starting directory. In this directory should be subdirectories corresponding to
        the 1,000 different categories of ImageNet.
    n_categories : int, optional
        Number of categories to process. Default is 0, meaning process all of them.
    n_samples_per_category : int, optional
        Number of files in each category to embed. Default is 0, meaning all files.

    Returns
    -------
    X: np.ndarray, shape=(-1, embedding_dimension)
        Image embeddings
    Y: np.ndarray, shape=(-1,)
        Synset category
    """
    category_dirs = list(input_dir.iterdir())
    if n_categories > 0:
        np.random.shuffle(category_dirs)
        category_dirs = category_dirs[:n_categories]
    X = []
    Y = []
    with ThreadPoolExecutor(max_workers=60) as executor:
        for i, category_dir in enumerate(category_dirs):
            futures = {}
            for j, image_file in enumerate(category_dir.iterdir()):
                if j == n_samples_per_category and n_samples_per_category > 0:
                    break
                image_bytes = image_file.read_bytes()
                if image_file.is_file():
                    future = executor.submit(
                        requests.post,
                        url="http://localhost:8000/v2/models/embed_image/infer",
                        data=image_bytes,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Inference-Header-Content-Length": "0",
                        },
                    )
                    futures[future] = image_file.name.split(".")[0]

            for future in as_completed(futures):
                try:
                    response = future.result()
                    label = futures[future].split("_")[0]
                except Exception as exc:
                    print(f"{futures[future]} threw {exc}")
                else:
                    try:
                        header_length = int(
                            response.headers["Inference-Header-Content-Length"]
                        )
                        embedding = np.frombuffer(
                            response.content[header_length:], dtype=np.float32
                        )
                        X.append(embedding)
                        Y.append(label)
                    except Exception as exc:
                        print(ValueError(f"Error getting data from response: {exc}"))
            if (i + 1) % 250 == 0:
                print(f"{(i+1):03} Finished {category_dir.name}")

        X = np.vstack(X)
        Y = np.array(Y)

    return X, Y


def zero_shot(labels_text: dict):
    """
    Get training vectors using a zero-shot approach. This will embed the text
    description of each of the synset categories using the siglip_text deployment
    using the prompt "This is a photo from ImageNet's {label} category. This
    category containes photos of {text}".

    Parameters
    ----------
    labels_text : dict
        Dictionary containing the synset category (e.g., n01742172) as the keys and
        their corresponding description(eg., 'boa constrictor, Constrictor constrictor')
        as the values

    Returns
    -------
    X: np.ndarray, shape=(n_categories, embedding_dimension)
        Text embeddings
    Y: np.ndarray, shape=(n_categories,)
        Synset category
    """
    X = []
    Y = []
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = {}
        for label, text in labels_text.items():
            prompt = f"A photo of a {text}."
            future = executor.submit(
                requests.post,
                url="http://localhost:8000/v2/models/embed_text/infer",
                json={
                    "inputs": [
                        {
                            "name": "INPUT_TEXT",
                            "shape": [1, 1],
                            "datatype": "BYTES",
                            "data": [prompt],
                        }
                    ]
                },
            )
            futures[future] = label

        for future in as_completed(futures):
            try:
                response = future.result()
                label = futures[future]
            except Exception as exc:
                print(f"{futures[future]} threw {exc}")
            try:
                embedding = response.json()["outputs"][0]["data"]
                embedding = np.array(embedding).astype(np.float32)
                X.append(embedding)
                Y.append(label)
            except Exception as exc:
                print(ValueError(f"Error getting json from response: {exc}"))

    X = np.vstack(X)
    Y = np.array(Y)

    return X, Y


def main():
    data_dir = Path(os.getenv("HOME")) / "data" / "imagenet"
    labels, labels_text = get_labels(data_dir / "LOC_synset_mapping.txt")
    valid_dir = data_dir / "valid"

    # Validation Data
    X, Y = get_image_embeddings(valid_dir)

    ##### Zero shot ######
    # Create training data using the text embedding of the label descriptions
    X_train_zero, Y_train_zero = zero_shot(labels_text)
    clf_zero = neighbors.KNeighborsClassifier(
        n_neighbors=10, weights="distance", metric="cosine"
    )
    clf_zero.fit(X_train_zero, Y_train_zero)

    # Make predictions
    Y_pred = clf_zero.predict_proba(X)

    # Results
    top1_accuracy = metrics.top_k_accuracy_score(Y, Y_pred, labels=labels, k=1)
    print(f"Zero-shot Top-1 Accuracy = {top1_accuracy:.4f}")

    top5_accuracy = metrics.top_k_accuracy_score(Y, Y_pred, labels=labels, k=5)
    print(f"Zero-shot Top-5 Accuracy = {top5_accuracy:.4f}")


if __name__ == "__main__":
    main()
