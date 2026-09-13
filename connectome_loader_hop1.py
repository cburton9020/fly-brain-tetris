"""
First-hop connectivity: flyvis-modeled cell types -> LC4 / LC6 / LPLC2.

We already have real neuPrint connections from LC4/LC6/LPLC2 to descending
neurons (visual_to_dn_connections.csv). But flyvis doesn't model LC4/LC6/
LPLC2 directly, so this script finds the real connections feeding INTO
those three types from cell types flyvis DOES model, completing the chain:

    flyvis-modeled cell (real activity available)
        -> (real synapse weights, this script)
    LC4 / LC6 / LPLC2
        -> (real synapse weights, connectome_loader.py)
    descending neuron
"""

import os

import pandas as pd
from dotenv import load_dotenv

from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies

# The full list of cell types flyvis's optic lobe model actually simulates,
# straight from list_flyvis_types.py's output.
FLYVIS_TYPES = [
    "C2", "C3",
    "L1", "L2", "L3", "L4", "L5",
    "Lawf1", "Lawf2",
    "Mi1", "Mi10", "Mi13", "Mi14", "Mi15", "Mi2", "Mi4", "Mi9",
    "T1", "T2", "T2a", "T3",
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm16", "Tm2", "Tm20", "Tm3", "Tm30", "Tm4",
    "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9",
    "TmY10", "TmY13", "TmY14", "TmY15", "TmY18", "TmY3", "TmY4", "TmY5a",
]

# These didn't come back with real matches in male-cns:v1.0 under this
# exact name (photoreceptors R1-R8, Am, Mi3/Mi11/Mi12, Tm28, TmY9, and
# both CT1 variants), left out rather than guessing alternate names.
CT1_TYPES = []

LC_TARGET_TYPES = ["LC4", "LC6", "LPLC2"]


def connect():
    load_dotenv()
    token = os.getenv("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("NEUPRINT_TOKEN not found in .env")
    return Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)


def fetch_flyvis_to_lc_connectivity(client):
    neuron_df, conn_df = fetch_adjacencies(
        sources=NC(type=FLYVIS_TYPES, regex=False),
        targets=NC(type=LC_TARGET_TYPES, regex=False),
        client=client,
    )

    if not CT1_TYPES:
        return neuron_df, conn_df

    try:
        ct1_neuron_df, ct1_conn_df = fetch_adjacencies(
            sources=NC(type=CT1_TYPES, regex=False),
            targets=NC(type=LC_TARGET_TYPES, regex=False),
            client=client,
        )
        combined_neuron_df = pd.concat([neuron_df, ct1_neuron_df]).drop_duplicates(subset="bodyId")
        combined_conn_df = pd.concat([conn_df, ct1_conn_df])
    except RuntimeError as e:
        print(f"(Skipping CT1(Lo1)/CT1(M10): {e})")
        combined_neuron_df, combined_conn_df = neuron_df, conn_df

    return combined_neuron_df, combined_conn_df


def summarize_and_save(neuron_df, conn_df):
    if conn_df.empty:
        print(
            "No direct connections found from any flyvis-modeled cell type "
            "to LC4, LC6, or LPLC2. This would mean the real path needs a "
            "third hop, an intermediate cell type between what flyvis "
            "models and these LC/LPLC neurons."
        )
        return

    named = conn_df.merge(
        neuron_df[["bodyId", "type"]].rename(columns={"bodyId": "bodyId_pre", "type": "type_pre"}),
        on="bodyId_pre", how="left",
    ).merge(
        neuron_df[["bodyId", "type"]].rename(columns={"bodyId": "bodyId_post", "type": "type_post"}),
        on="bodyId_post", how="left",
    )

    totals = named.groupby(["type_pre", "type_post"])["weight"].sum().sort_values(ascending=False)
    print(f"Found {len(conn_df)} connections, summarized by cell type pair:\n")
    print(totals.to_string())

    named.to_csv("flyvis_to_lc_connections.csv", index=False)
    print("\nSaved flyvis_to_lc_connections.csv")


if __name__ == "__main__":
    print("Connecting to neuPrint...")
    client = connect()

    print("Fetching connections from flyvis-modeled cell types to LC4/LC6/LPLC2...")
    neuron_df, conn_df = fetch_flyvis_to_lc_connectivity(client)

    summarize_and_save(neuron_df, conn_df)
