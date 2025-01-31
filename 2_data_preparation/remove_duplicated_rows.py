import pandas as pd

def main(df):
    # Contiamo le righe iniziali
    old_count = len(df)
    # Eliminiamo le righe duplicate
    duplicates_mask = df.duplicated(keep=False)
    # Contiamo le righe risultanti
    df = df[~duplicates_mask].copy()
    # Contiamo le righe risultanti
    new_count = len(df)
    print(f"Righe iniziali: {old_count}, Righe risultanti: {new_count}, Righe eliminate: {old_count - new_count}")
    # Reset degli indici
    df = df.reset_index(drop=True)
    df['ID'] = df.index + 1

    return df