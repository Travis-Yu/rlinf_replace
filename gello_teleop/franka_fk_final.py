"""Forward kinematics for the Franka Research 3 (Arm3R).

The kinematic chain is read from the official 'fr3.urdf' stored next to
'fr3.xml'. Poses are expressed from 'fr3_link0' to 'fr3_hand_tcp' by default.
Positions are returned in metres and quaternions use [x, y, z, w].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


_DEFAULT_URDF = Path(__file__).resolve().parent / "franka_fr3" / "fr3.urdf"
_ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _vector_attribute(
    element: ET.Element | None,
    attribute: str,
    default: str,
) -> np.ndarray:
    value = default if element is None else element.get(attribute, default)
    vector = np.fromstring(value, dtype=np.float64, sep=" ")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Expected three finite values for {attribute!r}, received {value!r}."
        )
    return vector


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        raise ValueError(f"Joint axis must be non-zero and finite, received {axis}.")

    x, y, z = axis / axis_norm
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer((x, y, z), (x, y, z))
        + sine * skew
    )


def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a deterministic unit quaternion."""
    quaternion = np.empty(4, dtype=np.float64)
    trace = np.trace(rotation)

    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion[3] = 0.25 * scale
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quaternion[3] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[0] = 0.25 * scale
        quaternion[1] = (rotation[0, 1] + rotation[1, 0]) / scale
        quaternion[2] = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quaternion[3] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[0] = (rotation[0, 1] + rotation[1, 0]) / scale
        quaternion[1] = 0.25 * scale
        quaternion[2] = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quaternion[3] = (rotation[1, 0] - rotation[0, 1]) / scale
        quaternion[0] = (rotation[0, 2] + rotation[2, 0]) / scale
        quaternion[1] = (rotation[1, 2] + rotation[2, 1]) / scale
        quaternion[2] = 0.25 * scale

    quaternion /= np.linalg.norm(quaternion)

    # q and -q are equivalent. Making the largest component positive is stable
    # even for the requested pose, whose scalar component is approximately zero.
    largest_component = int(np.argmax(np.abs(quaternion)))
    if quaternion[largest_component] < 0.0:
        quaternion = -quaternion
    return quaternion


def _rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis XYZ roll, pitch, yaw angles for diagnostic output."""
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-10:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


class FrankaFR3FK:
    """URDF-driven forward kinematics for the seven Arm3R joints."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        base_frame: str = "fr3_link0",
        end_frame: str = "fr3_hand_tcp",
    ) -> None:
        self.urdf_path = Path(urdf_path or _DEFAULT_URDF).expanduser().resolve()
        self.base_frame = base_frame
        self.end_frame = end_frame

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Arm3R URDF not found: {self.urdf_path}")

        try:
            root = ET.parse(self.urdf_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid URDF XML in {self.urdf_path}: {exc}") from exc

        self._chain = self._build_chain(root)
        moving_joint_names = tuple(
            joint.name for joint in self._chain if joint.joint_type != "fixed"
        )
        if moving_joint_names != _ARM_JOINT_NAMES:
            raise ValueError(
                "Expected the official seven-joint Arm3R chain "
                f"{_ARM_JOINT_NAMES}, received {moving_joint_names}."
            )

    def _build_chain(self, root: ET.Element) -> tuple[_Joint, ...]:
        links = {link.get("name") for link in root.findall("link")}
        for frame_name in (self.base_frame, self.end_frame):
            if frame_name not in links:
                raise ValueError(
                    f"Frame {frame_name!r} is absent from URDF {self.urdf_path}."
                )

        joint_by_child: dict[str, ET.Element] = {}
        for element in root.findall("joint"):
            child_element = element.find("child")
            if child_element is None or child_element.get("link") is None:
                raise ValueError(f"Joint {element.get('name')!r} has no child link.")
            child = child_element.get("link")
            if child in joint_by_child:
                raise ValueError(f"Multiple URDF joints have child link {child!r}.")
            joint_by_child[child] = element

        reversed_chain: list[_Joint] = []
        current_frame = self.end_frame
        visited: set[str] = set()

        while current_frame != self.base_frame:
            if current_frame in visited:
                raise ValueError(f"Cycle found while tracing URDF from {self.end_frame!r}.")
            visited.add(current_frame)

            element = joint_by_child.get(current_frame)
            if element is None:
                raise ValueError(
                    f"No URDF chain connects {self.base_frame!r} to "
                    f"{self.end_frame!r}; stopped at {current_frame!r}."
                )

            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or parent_element.get("link") is None:
                raise ValueError(f"Joint {element.get('name')!r} has no parent link.")

            joint_type = element.get("type", "")
            if joint_type not in {"fixed", "revolute", "continuous"}:
                raise ValueError(
                    f"Unsupported joint type {joint_type!r} in {element.get('name')!r}."
                )

            origin_element = element.find("origin")
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rotation_from_rpy(
                _vector_attribute(origin_element, "rpy", "0 0 0")
            )
            origin[:3, 3] = _vector_attribute(origin_element, "xyz", "0 0 0")

            axis = _vector_attribute(element.find("axis"), "xyz", "0 0 1")
            parent = parent_element.get("link")
            child = child_element.get("link")
            reversed_chain.append(
                _Joint(
                    name=element.get("name", ""),
                    joint_type=joint_type,
                    parent=parent,
                    child=child,
                    origin=origin,
                    axis=axis,
                )
            )
            current_frame = parent

        return tuple(reversed(reversed_chain))

    @staticmethod
    def _validate_joint_angles(joint_angles: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_angles, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(
                "joint_angles must have shape (7,) in radians; "
                f"received shape {q.shape}."
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("joint_angles contains NaN or infinity.")
        return q

    def forward_matrix(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return the 4x4 transform from base_frame to end_frame."""
        q = self._validate_joint_angles(joint_angles)
        transform = np.eye(4, dtype=np.float64)
        moving_index = 0

        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.joint_type in {"revolute", "continuous"}:
                joint_transform = np.eye(4, dtype=np.float64)
                joint_transform[:3, :3] = _axis_angle_rotation(
                    joint.axis,
                    q[moving_index],
                )
                transform = transform @ joint_transform
                moving_index += 1

        return transform

    def forward(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xyz_m, quaternion_xyzw) for seven joint angles in radians."""
        transform = self.forward_matrix(joint_angles)
        xyz_m = transform[:3, 3].copy()
        quaternion_xyzw = _quaternion_xyzw(transform[:3, :3])
        return xyz_m, quaternion_xyzw

    def get_fk(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility alias for callers that use the existing get_fk API."""
        return self.forward(joint_angles)

    def forward_pose7(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return [x, y, z, qx, qy, qz, qw] (metres, unit quaternion)."""
        xyz_m, quaternion_xyzw = self.forward(joint_angles)
        return np.concatenate((xyz_m, quaternion_xyzw))


# A convenient drop-in class name for code that imports FrankaFK.
FrankaFK = FrankaFR3FK


if __name__ == "__main__":
    audit_joint_angles = np.array(
        [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785],
        dtype=np.float64,
    )

    fk = FrankaFR3FK()
    audit_transform = fk.forward_matrix(audit_joint_angles)
    position_m, quaternion_xyzw = fk.forward(audit_joint_angles)

    print(f"URDF: {fk.urdf_path}")
    print(f"Transform: {fk.base_frame} -> {fk.end_frame}")
    print("Joint angles [rad]:", audit_joint_angles)
    print(
        "Position [m]:      ",
        np.array2string(position_m, precision=9, suppress_small=True),
    )
    print(
        "Position [mm]:     ",
        np.array2string(position_m * 1000.0, precision=3, suppress_small=True),
    )
    print(
        "RPY XYZ [deg]:     ",
        np.array2string(
            _rpy_degrees(audit_transform[:3, :3]),
            precision=3,
            suppress_small=True,
        ),
    )
    print("Quaternion [xyzw]: ", np.array2string(quaternion_xyzw, precision=9))
    print(f"Quaternion norm:    {np.linalg.norm(quaternion_xyzw):.12f}")
