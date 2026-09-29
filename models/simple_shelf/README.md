# Simple Shelf Model

This folder provides a minimal shelf with two compartments and no mesh dependencies.

## Files

- `simple_shelf.urdf`: URDF model using primitive boxes.
- `simple_shelf.sdf`: SDF model using primitive boxes.
- `model.config`: Gazebo model metadata.

## Dimensions

- Width: `0.8 m`
- Depth: `0.3 m`
- Height: `0.9 m`
- Board thickness: `0.02 m`
- Compartments: 2 stacked vertically

## PyBullet Usage (URDF)

```python
import pybullet as p
model_id = p.loadURDF("models/simple_shelf/simple_shelf.urdf", basePosition=[1.0, 0.0, 0.0], useFixedBase=True)
```

## Gazebo/SDF Usage

Use `simple_shelf.sdf` and `model.config` as a standard model folder.
