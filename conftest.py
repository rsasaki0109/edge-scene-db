"""Root conftest: disable ROS pytest plugins that interfere with standard tests."""

import sys


def pytest_configure(config):
    # Unregister ROS launch_testing plugins if loaded
    for name in list(config.pluginmanager.list_name_plugin()):
        plugin_name, plugin = name
        if "launch_testing" in str(plugin_name) or "launch_testing" in str(
            getattr(plugin, "__name__", "")
        ):
            try:
                config.pluginmanager.unregister(plugin, plugin_name)
            except Exception:
                pass
