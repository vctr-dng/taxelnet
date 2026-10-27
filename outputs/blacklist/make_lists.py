"""Split flagged_grasp_high_readings.csv (160 rows, a taxel > 3,000 TC) into glitch rows and the rest.

glitch = peak > 10,000 TC, or an isolated spike: peak > 3x its largest 4-neighbour with at most
2 taxels above 3,000 TC. Writes glitch_names.txt, all160_names.txt and flagged_classified.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/home/vctr-dng/robotic/taxelnet/data/newton/v1/training")
OUT = Path(__file__).parent
f = pd.read_csv(D / "flagged_grasp_high_readings.csv")
r = pd.read_csv(D / "Dataset_Grasp_Real_qtac100.csv").set_index("Name")
v = r.loc[f.Name, [f"tax{i}" for i in range(1, 29)]].to_numpy().reshape(-1, 7, 4)
iso = []
for m in v:
    i, j = np.unravel_index(m.argmax(), m.shape)
    nb = [
        m[a, b]
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        if 0 <= a < 7 and 0 <= b < 4
    ]
    iso.append(m[i, j] / max(max(nb), 1.0))
f["n_above_3000"] = (v > 3000).reshape(len(v), -1).sum(1)
f["isolation"] = np.round(iso, 2)
f["glitch"] = (f.peak_real > 10000) | ((f.isolation > 3) & (f.n_above_3000 <= 2))
f.to_csv(OUT / "flagged_classified.csv", index=False)
(OUT / "glitch_names.txt").write_text("\n".join(f.Name[f.glitch]) + "\n")
(OUT / "all160_names.txt").write_text("\n".join(f.Name) + "\n")
print(f.groupby(["sensor", "glitch"]).size(), "\nglitch total", int(f.glitch.sum()))
print("non-glitch peak", f[~f.glitch].peak_real.describe().round(0).to_dict())
