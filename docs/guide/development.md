# Loss development

Werner & Modlin §6, with the standard chain ladder treatment from Mack (1993).

## Chain ladder

```python
from pyratemaking.development import ChainLadder
import pandas as pd
import numpy as np

triangle = pd.DataFrame(
    {
        12: [357848, 352118, 290507, 310608],
        24: [1124788, 1236139, 1292306, np.nan],
        36: [1735330, 2170033, np.nan, np.nan],
        48: [2218270, np.nan, np.nan, np.nan],
    },
    index=[2018, 2019, 2020, 2021],
)
cl = ChainLadder(triangle, tail_factor=1.05)
cl.link_factors
cl.cdf
cl.ultimates()
```

Volume-weighted age-to-age factors by default (Mack 1993). Pass `weighted=False` for the simple-average variant.

## Bornhuetter-Ferguson

Blends an a priori ultimate (loss ratio × premium) with reported development:

```python
from pyratemaking.development import BornhuetterFerguson

a_priori = pd.Series({2018: 4_000_000, 2019: 4_200_000, 2020: 4_400_000, 2021: 4_600_000})
bf = BornhuetterFerguson(triangle, a_priori_ultimate=a_priori, tail_factor=1.05)
bf.ultimates()
```

## Cape Cod

Estimates the expected loss ratio from the data itself; useful when an a priori is not available:

```python
from pyratemaking.development import CapeCod

used_premium = pd.Series({2018: 5_500_000, 2019: 5_700_000, 2020: 5_900_000, 2021: 6_100_000})
cc = CapeCod(triangle, used_premium=used_premium, tail_factor=1.05, decay=0.10)
cc.expected_loss_ratio
cc.ultimates()
```

## Tail factors

Four candidates available:

```python
from pyratemaking.development.tail import (
    bondy_tail, sherman_tail, exponential_decay_tail, power_curve_tail, select_tail
)

select_tail(cl.link_factors, methods=("bondy", "sherman", "power", "exp_decay"))
```

## Bridge to `burncost`

When `pip install pyratemaking[burncost]` is satisfied, `ChainLadder.to_burncost()` hands the triangle off to that workflow without manual reshaping.
