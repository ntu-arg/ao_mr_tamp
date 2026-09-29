# Wooden Box Model

A lightweight wooden storage box model inspired by the IKEA UPPDATERA storage box.

## Target Outer Size

- Width: 0.25 m (25 cm)
- Depth: 0.35 m (35 cm)
- Height: 0.20 m (20 cm)

## Structure

- Bottom panel: 0.010 m thickness
- Side panels: 0.010 m thickness
- Front and back panels: 0.010 m thickness
- Top is open

## Files

- wooden_box.urdf: URDF model for PyBullet.
- wooden_box.sdf: SDF model for Gazebo/Ignition.
- model.config: model metadata.

## Notes

- The model is built from primitive boxes.
- Dimensions are approximate to the requested 25 x 35 x 20 cm size.

## PyBullet Example

```python
import pybullet as p
box_id = p.loadURDF(
    "models/wooden_box/wooden_box.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
```
