from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
from pathlib import Path
import requests
from sklearn import neighbors, metrics


def get_embeddings(input_dir: Path, n_categories: int = 0, max_workers: int = 60):
    embeddings = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, class_dir in enumerate(input_dir.iterdir()):
            # Stop if we only wanted to test a subset of the categories
            if i == n_categories and n_categories > 0:
                break
            futures = {}
            for image_file in class_dir.iterdir():
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
                        embeddings[futures[future]] = embedding
                    except Exception as exc:
                        print(ValueError(f"Error getting data from response: {exc}"))
            if (i + 1) % 100 == 0:
                print(f"{(i+1):03} Finished {class_dir.name}")

    return embeddings


def main():
    data_dir = Path(os.getenv("HOME")) / "data" / "imagenet"
    train_dir = data_dir / "train"
    train_embeddings = get_embeddings(train_dir)
    class_names = []
    class2id = {}
    X_train = []
    Y_train = []
    Z_train = []
    for filename, embedding in train_embeddings.items():
        class_name = filename.split("_")[0]
        if class_name not in class2id:
            class_names.append(class_name)
            class2id[class_name] = len(class_names) - 1
        X_train.append(embedding)
        Y_train.append(class2id[class_name])
        Z_train.append(class_name)

    print("Starting to fit KNN classifier")
    clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights="distance")
    clf.fit(X_train, Y_train)
    print("Finished fitting classifier")

    valid_dir = data_dir / "valid"
    valid_embeddings = get_embeddings(valid_dir)
    X = []
    Y_true = []
    Z_true = []
    for filename, embedding in valid_embeddings.items():
        class_name = filename.split("_")[0]
        X.append(embedding)
        Y_true.append(class2id[class_name])
        Z_true.append(class_name)
    Y_pred = clf.predict(X)

    average_precision = metrics.precision_score(Y_true, Y_pred, average="micro")
    print(f"{average_precision=:.4f}")


if __name__ == "__main__":
    main()
