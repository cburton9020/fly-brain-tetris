"""
Central brain / descending neuron connectivity loader.

Connects to neuPrint's public hemibrain connectome (the same underlying
dataset flyvis's optic lobe model is built from) and pulls two things:

  1. The real descending neurons (DNs), the cells that carry commands
     from the central brain down to the motor circuits controlling wings
     and legs. These are the natural candidates for our "motor readout"
     stage.
  2. Real, measured synaptic connection weights from a set of visual
     output cell types (the same ones flyvis exposes activity for, like
     T4c, LPLC2, LC4) to those descending neurons.

This gives us an actual measured adjacency matrix to sit between the eye
adapter's output and the small trainable readout layer, rather than
anything invented or randomly initialized.

Requires NEUPRINT_TOKEN in your .env file (see the setup steps from
earlier: sign in at neuprint.janelia.org, copy your Auth Token from your
account page).
"""

import os

import pandas as pd
from dotenv import load_dotenv

from neuprint import Client, NeuronCriteria as NC, fetch_neurons, fetch_adjacencies

# Visual output cell types we already get real activity for out of flyvis.
# T4/T5 subtypes are motion-direction-selective; LPLC2 and LC4 are known
# looming/collision-detecting projection neurons that are well studied for
# feeding into escape and steering circuits, good first candidates to look
# for direct or near-direct connections to descending neurons.
VISUAL_OUTPUT_TYPES = [
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
    "LPLC2", "LC4", "LC6",
]

# The regex below is a commonly used pattern (from neuPrint's own example
# notebooks) for catching most descending neuron naming conventions in the
# hemibrain dataset, plus the well known "Giant Fiber" escape neuron.
DN_TYPE_REGEX = r"(.*DN[^1]{0,}.*|Giant Fiber)"


def connect():
    load_dotenv()
    token = os.getenv("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "NEUPRINT_TOKEN not found. Make sure your .env file has a line "
            "like NEUPRINT_TOKEN=your_token_here."
        )
    return Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)


def fetch_descending_neurons(client):
    """Returns a DataFrame of real descending neurons in the hemibrain."""
    dns, _ = fetch_neurons(NC(type=DN_TYPE_REGEX, regex=True), client=client)
    return dns


def fetch_visual_to_dn_connectivity(client, dn_body_ids):
    """
    Returns (neuron_df, conn_df):
      neuron_df: properties of the neurons involved (bodyId, type, etc.)
      conn_df: bodyId_pre, bodyId_post, roi, weight for every connection
               found from our visual output types to the given DNs.
    """
    neuron_df, conn_df = fetch_adjacencies(
        sources=NC(type=VISUAL_OUTPUT_TYPES, regex=False),
        targets=NC(bodyId=list(dn_body_ids)),
        client=client,
    )
    return neuron_df, conn_df


def summarize(neuron_df, conn_df):
    """Prints a quick, human-readable summary of what was found."""
    if conn_df.empty:
        print(
            "No direct connections found from these visual output types to "
            "any descending neuron. This is a real, meaningful result, it "
            "likely means the path runs through one or more intermediate "
            "neurons rather than connecting directly. We'd need to add a "
            "multi-hop search to find that path."
        )
        return

    # Attach readable type names to both ends of each connection.
    named = conn_df.merge(
        neuron_df[["bodyId", "type"]].rename(
            columns={"bodyId": "bodyId_pre", "type": "type_pre"}
        ),
        on="bodyId_pre",
        how="left",
    ).merge(
        neuron_df[["bodyId", "type"]].rename(
            columns={"bodyId": "bodyId_post", "type": "type_post"}
        ),
        on="bodyId_post",
        how="left",
    )

    totals = (
        named.groupby(["type_pre", "type_post"])["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    print(f"Found {len(conn_df)} individual connections, summarized by cell type pair:\n")
    print(totals.to_string())


if __name__ == "__main__":
    print("Connecting to neuPrint (hemibrain dataset)...")
    client = connect()

    print("Fetching real descending neurons...")
    dns = fetch_descending_neurons(client)
    print(f"Found {len(dns)} descending neurons in the hemibrain dataset.")

    print("Fetching connectivity from visual output cells to descending neurons...")
    neuron_df, conn_df = fetch_visual_to_dn_connectivity(client, dns["bodyId"].values)

    summarize(neuron_df, conn_df)

    # Save the connection data WITH readable type names attached (not just
    # raw bodyIds), since that's what central_brain.py needs to build the
    # fixed weight table by cell type.
    named = conn_df.merge(
        neuron_df[["bodyId", "type"]].rename(
            columns={"bodyId": "bodyId_pre", "type": "type_pre"}
        ),
        on="bodyId_pre",
        how="left",
    ).merge(
        neuron_df[["bodyId", "type"]].rename(
            columns={"bodyId": "bodyId_post", "type": "type_post"}
        ),
        on="bodyId_post",
        how="left",
    )
    named.to_csv("visual_to_dn_connections.csv", index=False)
    dns.to_csv("descending_neurons.csv", index=False)
    print("\nSaved visual_to_dn_connections.csv and descending_neurons.csv")
