import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from implicit import als
from rich.console import Console
from rich.progress import track

console   = Console()
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models/als")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALS_CONFIG = {
    "factors":        64,
    "regularization": 0.01,
    "alpha":          40,
    "iterations":     20,
    "random_state":   42,
}


def load_data():
    console.print("[yellow]Loading processed data...[/yellow]")
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    val   = pd.read_parquet(PROC_DIR / "val.parquet")

    user2idx  = json.loads((PROC_DIR / "user2idx.json").read_text())
    movie2idx = json.loads((PROC_DIR / "movie2idx.json").read_text())

    n_users = len(user2idx)
    n_items = len(movie2idx)

    console.print(f"  Train: {len(train):,} | Users: {n_users:,} | Items: {n_items:,}")
    return train, val, n_users, n_items


def build_user_item_matrix(df, n_users, n_items):
    """
    Build user-item CSR matrix (rows=users, cols=items).
    Values = confidence: 1 + alpha * (rating / 5.0)
    """
    confidence = 1.0 + ALS_CONFIG["alpha"] * (df["rating"].values / 5.0)

    user_item = sp.csr_matrix(
        (confidence.astype(np.float32),
         (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_items),
    )
    console.print(f"  user-item matrix: {user_item.shape}, nnz={user_item.nnz:,}")
    return user_item


def train_als_model(user_item_matrix):
    """
    implicit v0.7+ API:
      model.fit() takes USER-ITEM matrix directly.
      After fit:
        model.user_factors → shape (n_users, factors)
        model.item_factors → shape (n_items, factors)
    """
    console.print(f"\n[yellow]Training ALS...[/yellow]")

    model = als.AlternatingLeastSquares(
        factors=ALS_CONFIG["factors"],
        regularization=ALS_CONFIG["regularization"],
        alpha=ALS_CONFIG["alpha"],
        iterations=ALS_CONFIG["iterations"],
        random_state=ALS_CONFIG["random_state"],
        use_gpu=False,
        calculate_training_loss=True,
    )

    t0 = time.time()
    model.fit(user_item_matrix)
    elapsed = time.time() - t0

    console.print(f"  [green]✓ Done in {elapsed:.1f}s[/green]")
    console.print(f"  user_factors: {model.user_factors.shape}")
    console.print(f"  item_factors: {model.item_factors.shape}")

    # Hard assertions — if these fail, the API has changed again
    assert model.user_factors.shape[0] == user_item_matrix.shape[0], (
        f"MISMATCH: user_factors[0]={model.user_factors.shape[0]} "
        f"!= n_users={user_item_matrix.shape[0]}"
    )
    assert model.item_factors.shape[0] == user_item_matrix.shape[1], (
        f"MISMATCH: item_factors[0]={model.item_factors.shape[0]} "
        f"!= n_items={user_item_matrix.shape[1]}"
    )
    console.print("  [green]✓ Shape assertions passed[/green]")
    return model


def evaluate(model, user_item_matrix, val, k=10, n_sample=1000):
    """Recall@K on sampled validation users."""
    console.print(f"\n[yellow]Evaluating Recall@{k}...[/yellow]")

    val_positives = (
        val[val["label"] == 1]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    eligible = [u for u in val_positives if u < user_item_matrix.shape[0]]
    sampled  = np.random.default_rng(42).choice(
        eligible, size=min(n_sample, len(eligible)), replace=False,
    )

    recalls = []
    for user_idx in track(sampled, description=f"Recall@{k}"):
        ground_truth = val_positives.get(int(user_idx), set())
        if not ground_truth:
            continue

        item_ids, _ = model.recommend(
            userid=int(user_idx),
            user_items=user_item_matrix[int(user_idx)],
            N=k,
            filter_already_liked_items=True,
        )

        hits   = len(set(item_ids.tolist()) & ground_truth)
        recalls.append(hits / len(ground_truth))

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    console.print(f"  [green]Recall@{k} = {mean_recall:.4f}[/green]")
    return mean_recall


def save(model, metrics):
    with open(MODEL_DIR / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    np.save(MODEL_DIR / "user_embeddings.npy", model.user_factors)
    np.save(MODEL_DIR / "item_embeddings.npy", model.item_factors)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  [green]✓ Saved → {MODEL_DIR}/[/green]")


def run():
    console.rule("[bold blue]ALS Baseline Training[/bold blue]")

    train, val, n_users, n_items = load_data()

    console.print("\n[yellow]Building user-item matrix...[/yellow]")
    user_item = build_user_item_matrix(train, n_users, n_items)

    model = train_als_model(user_item)

    metrics = {}
    for k in [10, 20, 50]:
        metrics[f"recall@{k}"] = evaluate(model, user_item, val, k=k)

    save(model, metrics)

    console.rule("[bold green]✓ ALS Baseline Complete[/bold green]")
    console.print("\n[bold]Baseline to beat with Two-Tower:[/bold]")
    for key, v in metrics.items():
        console.print(f"  {key:15} {v:.4f}")


if __name__ == "__main__":
    run()