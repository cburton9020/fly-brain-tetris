"""
Central brain connection layer, v2: combines two real pathways found in
the male-cns:v1.0 connectome data.

  1. DIRECT: some flyvis-modeled types (T4a-d, T5a-d) connect straight to
     certain descending neurons (found in visual_to_dn_connections.csv).
  2. TWO-HOP: many more flyvis-modeled types connect to LC4, LC6, or
     LPLC2 (found in flyvis_to_lc_connections.csv), which in turn connect
     to descending neurons (also in visual_to_dn_connections.csv).

Both pathways use only real, measured synapse counts as fixed weights.
Nothing here is trained or invented. The input to this layer is activity
for flyvis's own output cell types (whatever eye_adapter.py gives us),
since those are the only ones we actually have real activity values for.
"""

import pandas as pd


class CentralBrainLayer:
    def __init__(self, hop1_csv="flyvis_to_lc_connections.csv",
                 hop2_csv="visual_to_dn_connections.csv"):
        hop1_df = pd.read_csv(hop1_csv)   # flyvis type -> LC4/LC6/LPLC2
        hop2_df = pd.read_csv(hop2_csv)   # LC4/LC6/LPLC2 (or direct flyvis type) -> DN

        self.hop1 = hop1_df.groupby(["type_pre", "type_post"])["weight"].sum().reset_index()
        self.hop2 = hop2_df.groupby(["type_pre", "type_post"])["weight"].sum().reset_index()

        self.lc_types = sorted(self.hop1["type_post"].unique())
        self.dn_types = sorted(self.hop2["type_post"].unique())

        # Normalize hop1: for each LC type, what fraction of its modeled
        # input comes from each flyvis type.
        self._hop1_norm = self._normalize(self.hop1, group_col="type_post")
        # Normalize hop2: for each DN type, what fraction of its input
        # (from LC types AND any directly-connecting flyvis types) comes
        # from each source type.
        self._hop2_norm = self._normalize(self.hop2, group_col="type_post")

    @staticmethod
    def _normalize(df, group_col):
        totals = df.groupby(group_col)["weight"].transform("sum")
        out = df.copy()
        out["norm_weight"] = out["weight"] / totals
        return out

    def available_input_types(self):
        """All flyvis-modeled types that feed into this layer, directly
        or via LC4/LC6/LPLC2."""
        direct = set(self.hop2["type_pre"]) - set(self.lc_types)
        indirect = set(self.hop1["type_pre"])
        return sorted(direct | indirect)

    def compute_dn_activity(self, flyvis_activity):
        """
        flyvis_activity: dict mapping a flyvis cell type name (e.g. "T4c")
        to a scalar activity value.

        Returns a dict mapping each descending neuron type to its combined
        activity from both the direct and two-hop pathways, using real
        synapse-count-derived weights throughout.
        """
        # Step 1: propagate flyvis activity through hop1 to get an
        # estimated activity for LC4, LC6, and LPLC2.
        lc_activity = {}
        for lc_type in self.lc_types:
            rows = self._hop1_norm[self._hop1_norm["type_post"] == lc_type]
            total = 0.0
            for _, row in rows.iterrows():
                pre_type = row["type_pre"]
                if pre_type in flyvis_activity:
                    total += row["norm_weight"] * flyvis_activity[pre_type]
            lc_activity[lc_type] = total

        # Step 2: combine flyvis activity (direct pathway) and LC
        # activity (two-hop pathway) into descending neuron activity,
        # using hop2's normalized weights, which include both kinds of
        # source in the same table.
        combined_sources = dict(flyvis_activity)
        combined_sources.update(lc_activity)

        dn_activity = {}
        for dn_type in self.dn_types:
            rows = self._hop2_norm[self._hop2_norm["type_post"] == dn_type]
            total = 0.0
            for _, row in rows.iterrows():
                pre_type = row["type_pre"]
                if pre_type in combined_sources:
                    total += row["norm_weight"] * combined_sources[pre_type]
            dn_activity[dn_type] = total

        return dn_activity


if __name__ == "__main__":
    layer = CentralBrainLayer()
    print(f"{len(layer.available_input_types())} flyvis input types feed this layer "
          f"(directly or via {layer.lc_types}):")
    print(layer.available_input_types())
    print(f"\n{len(layer.dn_types)} descending neuron types reached.")

    # Smoke test: all inputs set to 1.0.
    fake_activity = {t: 1.0 for t in layer.available_input_types()}
    result = layer.compute_dn_activity(fake_activity)
    top = sorted(result.items(), key=lambda kv: -kv[1])[:10]
    print("\nTop 10 DN activities with all inputs set to 1.0:")
    for dn_type, value in top:
        print(f"  {dn_type}: {value:.4f}")
