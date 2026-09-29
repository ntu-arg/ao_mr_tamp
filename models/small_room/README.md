# Small Room Model

A simple open-top room asset with four walls for easy interior visibility.

## Dimensions

- Room size: `6m x 8m`
- Wall height: `0.3m`
- Wall thickness: `0.08m`
- Open top: yes

## Files

- `small_room.urdf`: PyBullet-friendly URDF using box primitives.
- `small_room.sdf`: SDF version.
- `model.config`: Gazebo metadata.

## Color

The room uses cool gray / blue tones so it is visually distinct from wooden objects.

## PyBullet example

```python
import pybullet as p
room_id = p.loadURDF("models/small_rooms/small_room.urdf", basePosition=[0, 0, 0], useFixedBase=True)
```
