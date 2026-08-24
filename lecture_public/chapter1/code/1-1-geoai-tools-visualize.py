"""
Create PNG snapshots from generated GeoJSON files.

Produces:
 - lecture_practice/chapter1/results/world_countries_110m.png
 - lecture_practice/chapter1/results/asia_countries.png

Run with the project's `.venv`:
  c:/.../.venv/Scripts/python.exe lecture_practice/chapter1/code/1-1-geoai-tools-visualize.py
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    world_geojson = results_dir / "world_countries_110m.geojson"
    asia_geojson = results_dir / "asia_countries.geojson"

    if not world_geojson.exists() or not asia_geojson.exists():
        print("GeoJSON files not found in results/. Run the preview script first.")
        return

    # World map snapshot
    world = gpd.read_file(world_geojson)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    world.plot(ax=ax, color="#e0e0e0", edgecolor="#333333", linewidth=0.25)
    ax.set_title("World countries (Natural Earth 110m)")
    ax.set_axis_off()
    out_world = results_dir / "world_countries_110m.png"
    fig.savefig(out_world, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_world}")

    # Asia map snapshot colored by POP_EST (if available)
    asia = gpd.read_file(asia_geojson)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    if "POP_EST" in asia.columns:
        asia.plot(column="POP_EST", ax=ax, cmap="viridis", edgecolor="#222222", linewidth=0.2, legend=True)
        ax.set_title("Asia countries — population estimate (POP_EST)")
    else:
        asia.plot(ax=ax, color="#99c2a2", edgecolor="#222222", linewidth=0.2)
        ax.set_title("Asia countries")
    ax.set_axis_off()
    out_asia = results_dir / "asia_countries.png"
    fig.savefig(out_asia, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_asia}")


if __name__ == "__main__":
    main()
