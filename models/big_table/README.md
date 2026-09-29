# Big Table Model

A separate rectangular table with twice the width and depth of `custom_table`.

## Outer Size

- Width: 3.20 m
- Depth: 1.60 m
- Height: 0.88 m

## Structural Dimensions

- Tabletop: 3.20 x 1.60 x 0.04 m
- Leg cross-section: 0.05 x 0.05 m
- Leg height: 0.84 m

The tabletop top surface is at 0.88 m when the model is loaded at z=0.

## PyBullet Example

```python
import pybullet as p

table_id = p.loadURDF(
    "models/big_table/big_table.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
```
