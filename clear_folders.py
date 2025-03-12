import os
import shutil
import configparser

# Konfiguration aus der config.ini laden
config = configparser.ConfigParser()
config.read('config.ini')

# Pfade aus der config.ini auslesen
HORNS_FOLDER = config['Paths']['HORNS_FOLDER']
CONFIGS_FOLDER = config['Paths']['CONFIGS_FOLDER']

# Datenbank-Dateipfad (anpassen, falls nötig)
DATABASE_FILE = "waveguides.db"

# Funktion zum Leeren eines Ordners
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Löscht den Ordner und alle Inhalte
            else:
                os.remove(item_path)  # Löscht die Datei
        print(f"Ordner geleert: {folder_path}")
    else:
        print(f"Ordner existiert nicht: {folder_path}")

# Funktion zum Löschen der Datenbankdatei
def clear_database(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Datenbankdatei gelöscht: {db_path}")
    else:
        print(f"Datenbankdatei existiert nicht: {db_path}")

# Hauptausführung
if __name__ == "__main__":
    confirmation = input("To delete all simulations, configs, and the database, type 'YES': ")
    if confirmation == "YES":
        clear_folder(HORNS_FOLDER)
        clear_folder(CONFIGS_FOLDER)
        clear_database(DATABASE_FILE)
    else:
        print("Operation abgebrochen.")
