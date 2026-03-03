"""
Alarm Utility Module
Provides server-side alarm functionality.
Primary alerts happen on the frontend via Web Audio API.
"""

import threading
import os
import sys


class AlarmManager:
    """Manages server-side alarm sounds."""

    def __init__(self):
        self.is_playing = False
        self._lock = threading.Lock()
        self.alarm_file = os.path.join(
            os.path.dirname(__file__), "..", "alarms", "alarm.wav"
        )

    def play_alarm(self):
        """Play alarm sound in a separate thread (Windows only)."""
        if sys.platform == "win32":
            with self._lock:
                if self.is_playing:
                    return
                self.is_playing = True

            def _play():
                try:
                    import winsound
                    # Play a beep sound (frequency=2500Hz, duration=1000ms)
                    winsound.Beep(2500, 1000)
                except Exception:
                    pass
                finally:
                    with self._lock:
                        self.is_playing = False

            thread = threading.Thread(target=_play, daemon=True)
            thread.start()

    def stop_alarm(self):
        """Stop the alarm."""
        with self._lock:
            self.is_playing = False
