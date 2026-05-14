# data/preprocess.py
"""
Transform raw MovieLens data into training-ready format.

Three key decisions to know cold for interviews:
  1. Implicit feedback: rating >= 4.0 → positive label
  2. Temporal split: split by TIME not randomly (prevents data leakage)
  3. Dense ID mapping: sparse userId → sequential index for embedding layers
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

RAW_DIR  = Path("data/raw/ml-25m")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    console.print("[yellow]Loading ratings.csv...[/yellow]")
    ratings = pd.read_csv(
        RAW_DIR / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
    )
    console.print("[yellow]Loading movies.csv...[/yellow]")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    return ratings, movies


def make_implicit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    rating >= 4.0 → label=1 (positive)
    rating <  4.0 → label=0 (negative)

    Also extract temporal context features — useful for the Two-Tower model.
    """
    df = df.copy()
    df["label"]       = (df["rating"] >= 4.0).astype("int8")
    df["datetime"]    = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = df["datetime"].dt.hour.astype("int8")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("int8")
    df["month"]       = df["datetime"].dt.month.astype("int8")
    df.drop(columns=["datetime"], inplace=True)
    return df


def build_id_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    PyTorch Embedding(n_users, embed_dim) needs user IDs in [0, n_users).
    MovieLens user IDs go up to 162541 — not sequential after filtering.
    We remap to dense sequential integers.
    """
    user_ids  = sorted(df["userId"].unique())
    movie_ids = sorted(df["movieId"].unique())

    user2idx  = {int(u): i for i, u in enumerate(user_ids)}
    movie2idx = {int(m): i for i, m in enumerate(movie_ids)}

    df["user_idx"]  = df["userId"].map(user2idx).astype("int32")
    df["movie_idx"] = df["movieId"].map(movie2idx).astype("int32")

    return df, user2idx, movie2idx


def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by timestamp → take last (val_frac + test_frac) as holdout.

    Why this matters (interview answer):
      Random split lets the model train on a user's 2023 ratings to
      predict their 2021 ratings — a future-data leak. Temporal split
      ensures we only predict FORWARD in time, matching production.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)
    i1 = int(n * (1 - val_frac - test_frac))
    i2 = int(n * (1 - test_frac))

    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def compute_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user behaviour — computed on TRAIN ONLY to avoid leakage."""
    feats = train.groupby("user_idx").agg(
        n_interactions  = ("label",     "count"),
        positive_rate   = ("label",     "mean"),
        avg_rating      = ("rating",    "mean"),
        first_ts        = ("timestamp", "min"),
        last_ts         = ("timestamp", "max"),
        n_unique_movies = ("movie_idx", "nunique"),
    ).reset_index()

    feats["activity_span_days"] = (
        (feats["last_ts"] - feats["first_ts"]) / 86400
    ).round(1)
    return feats


def compute_item_features(train: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item statistics — train only."""
    feats = train.groupby("movie_idx").agg(
        n_ratings      = ("label",     "count"),
        positive_rate  = ("label",     "mean"),
        avg_rating     = ("rating",    "mean"),
        first_rated_ts = ("timestamp", "min"),
        last_rated_ts  = ("timestamp", "max"),
    ).reset_index()

    # Add genre binary features (top genres as one-hot columns)
    top_genres = ["Action", "Comedy", "Drama", "Thriller",
                  "Romance", "Horror", "Animation", "Documentary"]
    for g in top_genres:
        movies[f"genre_{g.lower()}"] = (
            movies["genres"].str.contains(g, na=False).astype("int8")
        )
    return feats, movies


def print_summary(train, val, test, user2idx, movie2idx):
    table = Table(title="Preprocessing Summary", show_header=True)
    table.add_column("Split",   style="bold")
    table.add_column("Rows",    justify="right")
    table.add_column("% Total", justify="right")

    n = len(train) + len(val) + len(test)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        table.add_row(name, f"{len(df):,}", f"{len(df)/n:.1%}")

    console.print(table)
    console.print(f"\n  n_users : {len(user2idx):,}")
    console.print(f"  n_items : {len(movie2idx):,}")
    console.print(f"  +labels : {train['label'].sum():,} "
                  f"({train['label'].mean():.1%} of train)")


def run():
    console.rule("[bold blue]MovieLens Preprocessing[/bold blue]")

    ratings, movies = load_raw()

    console.print("\n[yellow]Creating implicit labels + temporal features...[/yellow]")
    df = make_implicit_labels(ratings)

    console.print("[yellow]Building dense ID mappings...[/yellow]")
    df, user2idx, movie2idx = build_id_maps(df)

    console.print("[yellow]Temporal train / val / test split...[/yellow]")
    train, val, test = temporal_split(df)

    console.print("[yellow]Computing user & item features (train only)...[/yellow]")
    user_feats            = compute_user_features(train)
    item_feats, movies_clean = compute_item_features(train, movies)

    console.print("[yellow]Saving parquet files...[/yellow]")
    train.to_parquet(PROC_DIR / "train.parquet",              index=False)
    val.to_parquet(PROC_DIR / "val.parquet",                  index=False)
    test.to_parquet(PROC_DIR / "test.parquet",                index=False)
    user_feats.to_parquet(PROC_DIR / "user_features.parquet", index=False)
    item_feats.to_parquet(PROC_DIR / "item_features.parquet", index=False)
    movies_clean.to_parquet(PROC_DIR / "movies_clean.parquet",index=False)

    # Save ID mappings as JSON (needed by serving layer later)
    (PROC_DIR / "user2idx.json").write_text(
        json.dumps(user2idx, indent=2)
    )
    (PROC_DIR / "movie2idx.json").write_text(
        json.dumps(movie2idx, indent=2)
    )

    print_summary(train, val, test, user2idx, movie2idx)
    console.rule("[bold green]✓ Done — data/processed/ is ready[/bold green]")
    console.print("\n[bold]Next:[/bold] python data/preprocess.py passed → "
                  "open notebooks/01_eda.ipynb")


if __name__ == "__main__":
    run()