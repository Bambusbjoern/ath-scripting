import sqlite3
import matplotlib.pyplot as plt
import numpy as np

DB_PATH = "waveguides.db"


def visualize_ratings_distribution(db_path=DB_PATH):
    """
    Connects to the given waveguides database, retrieves all rating values,
    and shows a histogram of the rating distribution (0 to 50).
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT rating
        FROM waveguide_params
        WHERE rating IS NOT NULL
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        print("No ratings found in the database.")
        return

    ratings = np.array([r[0] for r in rows], dtype=float)

    plt.figure(figsize=(8, 6))

    # Specify 'range=(0, 50)' so bins are allocated only within this interval,
    # effectively “zooming in” on that region in the histogram itself.
    plt.hist(ratings, bins=127, range=(5, 50),
             color='steelblue', edgecolor='black', alpha=0.7)

    plt.title("Waveguide Rating Distribution (0–50)")
    plt.xlabel("Rating")
    plt.ylabel("Count of Waveguides")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    visualize_ratings_distribution()
