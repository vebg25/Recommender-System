# data/download.py
"""
Download MovieLens 25M dataset.
25 million ratings, 62K movies, 162K users — realistic scale for interviews.
"""
import urllib.request
import zipfile
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_with_progress(url: str, dest: Path):
    """Download with a real progress bar."""
    with Progress(
        "[progress.description]{task.description}",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Downloading MovieLens 25M...", total=None)

        def hook(block_num, block_size, total_size):
            if total_size > 0:
                progress.update(task, total=total_size,
                                completed=min(block_num * block_size, total_size))

        urllib.request.urlretrieve(url, dest, reporthook=hook)


def download_movielens() -> Path:
    zip_path     = RAW_DIR / "ml-25m.zip"
    extract_path = RAW_DIR / "ml-25m"

    if extract_path.exists() and any(extract_path.iterdir()):
        console.print("[green]✓ MovieLens 25M already downloaded[/green]")
        return extract_path

    console.print(f"\n[bold]Downloading MovieLens 25M[/bold]")
    console.print(f"Source: {MOVIELENS_URL}")
    console.print(f"Destination: {zip_path.resolve()}\n")

    download_with_progress(MOVIELENS_URL, zip_path)
    console.print("[green]✓ Download complete[/green]")

    console.print("\n[yellow]Extracting zip...[/yellow]")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    console.print("[green]✓ Extracted[/green]")

    zip_path.unlink()   # Delete zip to save disk space
    return extract_path


def verify(data_dir: Path):
    files = {
        "ratings.csv": "25M ratings",
        "movies.csv":  "62K movies",
        "tags.csv":    "1M tags",
        "links.csv":   "Movie ID links",
    }
    console.print("\n[bold]Verifying files:[/bold]")
    all_ok = True
    for fname, desc in files.items():
        fpath = data_dir / fname
        if fpath.exists():
            mb = fpath.stat().st_size / 1e6
            console.print(f"  [green]✓[/green] {fname:<20} {mb:6.1f} MB  ({desc})")
        else:
            console.print(f"  [red]✗ MISSING: {fname}[/red]")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    console.rule("[bold blue]MovieLens 25M Downloader[/bold blue]")
    data_dir = download_movielens()
    ok = verify(data_dir)

    if ok:
        console.print("\n[bold green]✓ Dataset ready![/bold green]")
        console.print("[bold]Next step:[/bold] python data/preprocess.py")
    else:
        console.print("\n[red]Some files missing — try running again[/red]")