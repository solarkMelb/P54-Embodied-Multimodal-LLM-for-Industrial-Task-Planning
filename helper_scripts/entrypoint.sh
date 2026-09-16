#!/bin/bash
# Start virtual display for PyBullet GUI
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
exec "$@"