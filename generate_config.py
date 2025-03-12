import os
from database_helper import get_params_by_id  # Importiere die Funktion, die die Parameter abruft

def generate_waveguide_config(configs_folder, filename, config_id, verbose=False):
    """
    Erzeugt eine Konfigurationsdatei, indem die Parameter aus der Datenbank anhand der ID abgerufen werden.

    Parameter:
    - configs_folder: Pfad, wo die Konfigurationsdateien gespeichert werden.
    - filename: Dateiname, z.B. "123.cfg", basierend auf der config_id.
    - config_id: Die ID, unter der die Parameter in der Datenbank gespeichert sind.
    - verbose: Wenn True, werden detaillierte Informationen ausgegeben.
    """
    try:
        # Rufe die Parameter aus der Datenbank ab
        params = get_params_by_id(config_id, db_path="waveguides.db")
        if not params:
            if verbose:
                print(f"Keine Parameter in der DB für ID={config_id}.")
            return False

        # Entpacke die Parameter aus dem Dictionary
        r0 = params["r0"]
        a0 = params["a0"]
        a = params["a"]
        k = params["k"]
        L = params["L"]
        s = params["s"]
        n = params["n"]
        q = params["q"]
        u_va = params["u_va"]
        u_va0 = params["u_va0"]
        u_vk = params["u_vk"]
        u_vs = params["u_vs"]
        u_vn = params["u_vn"]
        mfp = params["mfp"]
        mr = params["mr"]

        # Stelle sicher, dass der Ordner existiert, oder erstelle ihn
        if not os.path.exists(configs_folder):
            os.makedirs(configs_folder)
            if verbose:
                print(f"Created directory: {configs_folder}")

        # Lade das Basis-Template für die Konfigurationsdatei
        with open('base_template.txt', 'r') as file:
            base_template_content = file.read()

        # Berechne ggf. abgeleitete Parameter, z.B.:
        va = round(-(20-a + u_va * (80-20)), 2)
        va0 = round(-(20 - a0 + u_va0 * (80-20)), 2)  # Beispiel: abgeleiteter Parameter für va0
        vk = round(-(0.1 - k + u_vk * 9.9), 2)  # Beispiel: abgeleiteter Parameter für vk
        vs = round(-(-s + u_vs * 2), 2)  # Beispiel: abgeleiteter Parameter für vs
        vn = round(-(0 - n + u_vn * 10), 2)  # Beispiel: abgeleiteter Parameter für vn

        # Fülle das Template mit den Werten
        config_content = base_template_content.format(
            r0=r0, a0=a0, a=a, k=k, L=L, s=s, n=n, q=q,
            va=va, va0=va0, vk=vk, vs=vs, vn=vn, mfp=mfp, mr=mr
        )

        # Schreibe den Inhalt in die Konfigurationsdatei
        filepath = os.path.join(configs_folder, filename)
        with open(filepath, 'w') as config_file:
            config_file.write(config_content)

        if verbose:
            print(f"Created configuration file: {filepath}")

        return True
    except Exception as e:
        if verbose:
            print(f"Failed to create configuration file: {e}")
        return False
