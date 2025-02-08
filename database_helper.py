import sqlite3  # Importiere das Modul für SQLite-Datenbanken


def initialize_db(db_path="waveguides.db"):
    """
    Diese Funktion stellt eine Verbindung zu einer SQLite-Datenbank her und erstellt
    eine Tabelle namens 'waveguide_params', falls sie noch nicht existiert.

    Parameter:
    - db_path: Der Pfad zur Datenbankdatei (Standard: "waveguides.db").
    """
    # Verbindung zur Datenbank herstellen. Existiert die Datei noch nicht, wird sie erstellt.
    conn = sqlite3.connect(db_path)
    # Ein Cursor-Objekt ermöglicht das Ausführen von SQL-Befehlen.
    c = conn.cursor()

    # SQL-Befehl zum Erstellen der Tabelle, falls sie noch nicht existiert.
    c.execute("""
        CREATE TABLE IF NOT EXISTS waveguide_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Automatisch fortlaufende ID
            r0 REAL,      -- Parameter r0 als Fließkommazahl
            a0 REAL,      -- Parameter a0
            a REAL,       -- Parameter a
            k REAL,       -- Parameter k
            L REAL,       -- Parameter L
            s REAL,       -- Parameter s
            n REAL,       -- Parameter n
            q REAL,       -- Parameter q
            va REAL,      -- Parameter va
            u_va0 REAL,   -- Parameter u_va0 (normalisierter Wert)
            u_vk REAL,    -- Parameter u_vk
            u_vs REAL,    -- Parameter u_vs
            u_vn REAL,    -- Parameter u_vn
            mfp REAL,     -- Parameter mfp
            mr REAL,
            rating REAL  -- Neue Spalte zum Speichern des Simulationsergebnisses
        )
    """)
    conn.commit()
    conn.close()


def insert_params(param_values, db_path="waveguides.db"):
    """
    Diese Funktion fügt einen neuen Satz von Parametern in die Datenbank ein.

    Parameter:
    - param_values: Ein Dictionary, das die Parameter enthält. Die Keys sollten
      u.a. "r0", "a0", "a", "k", "L", "s", "n", "q", "va", "u_va0", "u_vk", "u_vs",
      "u_vn", "mfp" und "mr" heißen.
    - db_path: Der Pfad zur Datenbankdatei (Standard: "waveguides.db").

    Rückgabe:
    - Die automatisch generierte ID des neu eingefügten Datensatzes.
    """
    # Verbindung zur Datenbank herstellen
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # SQL-Befehl, um einen neuen Datensatz in die Tabelle einzufügen.
    c.execute("""
        INSERT INTO waveguide_params
        (r0, a0, a, k, L, s, n, q, va, u_va0, u_vk, u_vs, u_vn, mfp, mr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        param_values["r0"],
        param_values["a0"],
        param_values["a"],
        param_values["k"],
        param_values["L"],
        param_values["s"],
        param_values["n"],
        param_values["q"],
        param_values["va"],
        param_values["u_va0"],
        param_values["u_vk"],
        param_values["u_vs"],
        param_values["u_vn"],
        param_values["mfp"],
        param_values["mr"]
    ))

    # Die ID des neu eingefügten Datensatzes abrufen
    new_id = c.lastrowid
    conn.commit()  # Änderungen speichern
    conn.close()  # Verbindung schließen

    return new_id  # Neue ID zurückgeben


def get_params_by_id(config_id, db_path="waveguides.db"):
    """
    Diese Funktion ruft die Parameter zu einem bestimmten Datensatz (über die ID)
    aus der Datenbank ab.

    Parameter:
    - config_id: Die ID des gewünschten Datensatzes.
    - db_path: Der Pfad zur Datenbankdatei (Standard: "waveguides.db").

    Rückgabe:
    - Ein Dictionary mit den Parametern, oder None, wenn der Datensatz nicht gefunden wurde.
    """
    # Verbindung zur Datenbank herstellen
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # SQL-Befehl, um den Datensatz mit der angegebenen ID auszuwählen.
    c.execute("""
        SELECT r0, a0, a, k, L, s, n, q, va, u_va0, u_vk, u_vs, u_vn, mfp, mr
        FROM waveguide_params WHERE id=?
    """, (config_id,))

    # Ein einzelnes Ergebnis abrufen
    row = c.fetchone()
    conn.close()  # Verbindung schließen

    if not row:
        return None  # Wenn kein Datensatz gefunden wurde, None zurückgeben

    # Erstelle ein Dictionary, das die Parameter mit ihren Namen enthält.
    keys = ["r0", "a0", "a", "k", "L", "s", "n", "q", "va", "u_va0", "u_vk", "u_vs", "u_vn", "mfp", "mr"]
    return dict(zip(keys, row))


def get_completed_simulations(db_path="waveguides.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT r0, a0, a, k, L, s, n, q, va, u_va0, u_vk, u_vs, u_vn, mfp, mr, rating
        FROM waveguide_params
        WHERE rating IS NOT NULL
    """)
    rows = c.fetchall()
    conn.close()
    # Define keys in the order stored:
    keys = ["r0", "a0", "a", "k", "L", "s", "n", "q", "va", "u_va0", "u_vk", "u_vs", "u_vn", "mfp", "mr"]
    simulations = []
    ratings = []
    for row in rows:
        sim = dict(zip(keys, row[:15]))
        simulations.append(sim)
        ratings.append(row[15])
    return simulations, ratings


def update_rating(config_id, rating, db_path="waveguides.db"):
    """
    Aktualisiert den Rating-Wert für den Datensatz mit der angegebenen ID.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE waveguide_params SET rating=? WHERE id=?", (rating, config_id))
    conn.commit()
    conn.close()
