import sqlite3

DB_PATH = "waveguides.db"

def list_five_lowest_ratings(db_path=DB_PATH):
    """
    Lists the five waveguides with the lowest rating from the 'waveguide_params' table.
    Prints the ID and rating for each.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Assuming 'rating' is stored in the 'waveguide_params' table
    c.execute("""
        SELECT id, rating
        FROM waveguide_params
        WHERE rating IS NOT NULL
        ORDER BY rating ASC
        LIMIT 5
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        print("No rated waveguides found in the database.")
        return

    print("=== Five Waveguides with Lowest Ratings ===")
    for idx, (wg_id, wg_rating) in enumerate(rows, start=1):
        print(f"{idx}. ID={wg_id}, Rating={wg_rating}")

if __name__ == "__main__":
    list_five_lowest_ratings()
