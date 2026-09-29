# High Shelf Model

A lightweight high-shelf model built from primitive boxes.

## Target Outer Size

- Width: `0.60 m`
- Depth: `0.27 m`
- Height: `1.40 m`

## Files

- `high_shelf.urdf`: URDF model for PyBullet.
- `high_shelf.sdf`: SDF model for Gazebo/ignition.
- `model.config`: model metadata.

## Notes

- This is an approximate open-frame shelf style model sized to 60x27x140 cm.
- The model uses 4 vertical legs and 4 shelf boards.

## Shelf Level Heights

All heights below are measured from the floor (`z=0`) when loaded with `basePosition=[0, 0, 0]`.

- Shelf board thickness: `0.015 m` (1.5 cm)

| Level | Joint/Center z (m) | Top Surface z (m) | Underside z (m) |
|---|---:|---:|---:|
| Shelf 1 (bottom) | 0.0375 | 0.0450 | 0.0300 |
| Shelf 2 | 0.4875 | 0.4950 | 0.4800 |
| Shelf 3 | 0.9375 | 0.9450 | 0.9300 |
| Shelf 4 (top) | 1.3925 | 1.4000 | 1.3850 |

- Vertical spacing between shelf top surfaces is `0.45 m` (45 cm).

## PyBullet

```python
import pybullet as p
shelf_id = p.loadURDF(
    "models/high_shelf/high_shelf.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
```
