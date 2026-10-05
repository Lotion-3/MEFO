"""SYNTHETIC dataset on the toy network: targets are the analytic FBA optimum.

It exists only to test the pipeline end to end. FBA scoring zero error here is true by
construction and says nothing about FBA's real-world accuracy.
"""

from __future__ import annotations

import pandas as pd

from metabench.core.interfaces import Dataset, StandardizedData
from metabench.core.registry import register_dataset
from metabench.metabolic.toy import analytic_solution

DEFAULT_UPTAKES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


@register_dataset("toy")
class ToyDataset(Dataset):
    def load(self) -> StandardizedData:
        uptakes = [float(u) for u in self.params.get("uptakes", DEFAULT_UPTAKES)]
        cond_ids = [f"u{u:g}" for u in uptakes]
        conditions = pd.DataFrame(
            {
                "condition_id": cond_ids,
                "organism": "toy",
                "model_id": "toy",
                "carbon_source": "A",
                # Two coarse regimes, handy for testing leave-group-out on a non-id column.
                "regime": ["low" if u <= 4 else "high" for u in uptakes],
            }
        )
        rows = []
        for cid, u in zip(cond_ids, uptakes, strict=True):
            sol = analytic_solution(u)
            rows.append((cid, "growth", "growth", sol["BIO"], "1/h"))
            rows.append((cid, "flux", "R2", sol["R2"], "mmol/gDW/h"))
            rows.append((cid, "flux", "R3", sol["R3"], "mmol/gDW/h"))
        measurements = pd.DataFrame(
            rows, columns=["condition_id", "measurement_type", "target_id", "value", "units"]
        )
        features = pd.DataFrame(
            {"uptake_A": uptakes}, index=pd.Index(cond_ids, name="condition_id")
        )
        return StandardizedData(
            name="toy",
            conditions=conditions,
            measurements=measurements,
            media={cid: {"EX_A": u} for cid, u in zip(cond_ids, uptakes, strict=True)},
            features=features,
            source="synthetic: analytic optimum of metabench.metabolic.toy",
            citation="n/a (synthetic)",
            license="n/a (synthetic)",
            synthetic=True,
        )
