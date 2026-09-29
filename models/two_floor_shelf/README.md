# Two-Floor Shelf Model

An open two-floor shelf built from primitive boxes.

## Target Outer Size

- Width: 0.31 m (31 cm)
- Depth: 0.18 m (18 cm)
- Height: 0.42 m (42 cm)

## Structure

- Top panel: 0.01 m thickness
- Bottom panel: 0.01 m thickness
- Middle divider (second floor): 0.01 m thickness
- Four vertical corner struts: 0.01 x 0.01 x 0.42 m
- Side panels: none
- Back panel: none

## Floor Heights

All heights are measured from the floor (`z=0`) when loaded with `basePosition=[0, 0, 0]`.

- Second floor (middle divider) center height: `0.21 m` (21 cm)
- Second floor top surface height: `0.215 m` (21.5 cm)
- Second floor underside height: `0.205 m` (20.5 cm)
- Top surface height: `0.42 m` (42 cm)

## Files

- two_floor_shelf.urdf: URDF for PyBullet
- two_floor_shelf.sdf: SDF for Gazebo/Ignition
- model.config: model metadata

## PyBullet Example

```python
import pybullet as p
shelf_id = p.loadURDF(
    "models/two_floor_shelf/two_floor_shelf.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
```
