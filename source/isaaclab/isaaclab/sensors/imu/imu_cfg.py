# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .imu import Imu


@configclass
class ImuCfg(SensorBaseCfg):
    """Configuration for an Inertial Measurement Unit (IMU) sensor."""

    class_type: type = Imu

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration object for the visualization markers. Defaults to RED_ARROW_X_MARKER_CFG.

    This attribute is only used when debug visualization is enabled.
    """
    gravity_bias: tuple[float, float, float] = (0.0, 0.0, 9.81)
    """The linear acceleration bias applied to the linear acceleration in the world frame (x,y,z).

    Imu sensors typically output a positive gravity acceleration in opposition to the direction of gravity. This
    config parameter allows users to subtract that bias if set to (0.,0.,0.). By default this is set to (0.0,0.0,9.81)
    which results in a positive acceleration reading in the world Z.
    """

    max_linear_velocity: float = 100.0
    """Maximum linear velocity magnitude (m/s) before clipping. Defaults to 100.0 m/s."""

    max_angular_velocity: float = 50.0
    """Maximum angular velocity magnitude (rad/s) before clipping. Defaults to 50.0 rad/s."""

    max_linear_acceleration: float = 1000.0
    """Maximum linear acceleration magnitude (m/s²) before clipping. Defaults to 1000.0 m/s²."""

    max_angular_acceleration: float = 1000.0
    """Maximum angular acceleration magnitude (rad/s²) before clipping. Defaults to 1000.0 rad/s²."""

    min_dt: float = 1e-6
    """Minimum timestep for numerical differentiation to prevent division by zero. Defaults to 1e-6 seconds."""
