import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

class RadarDisplay:
    """
    Polar Plot (PPI) Display for Passive Radar.
    Shows Range (radius) vs Azimuth (angle).

    This runs in a separate thread/process usually, or updated via FuncAnimation.
    """
    def __init__(self, max_range=100, update_interval=500):
        self.max_range = max_range
        self.interval = update_interval

        # Detections: List of (azimuth_deg, range_bin, power)
        self.detections = []
        self.lock = threading.Lock()
        self.running = False

        self.fig = None
        self.ax = None
        self.scat = None

    def start(self):
        """Starts the display loop (blocking or threaded)"""
        # We assume this is called in main thread if using matplotlib backend
        self.running = True
        self._setup_plot()
        plt.show() # Blocking

    def _setup_plot(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1) # Clockwise
        self.ax.set_ylim(0, self.max_range)
        self.ax.set_title("KrakenSDR Passive Radar PPI")

        # Scatter plot for targets
        self.scat = self.ax.scatter([], [], c=[], cmap='autumn', s=50, alpha=0.8)

        self.anim = FuncAnimation(self.fig, self._update, interval=self.interval, blit=False)

    def _update(self, frame):
        with self.lock:
            dets = self.detections[:] # copy

        if not dets:
            self.scat.set_offsets(np.empty((0, 2)))
            return self.scat,

        # Convert to polar coordinates for Scatter
        # Matplotlib polar expects (theta_radians, r)
        # Our azimuth is degrees from North (0).
        # set_theta_zero_location('N') handles the 0=North.

        angles = [np.radians(d[0]) for d in dets]
        ranges = [d[1] for d in dets]
        powers = [d[2] for d in dets]

        # Colors based on power?

        data = np.column_stack((angles, ranges))
        self.scat.set_offsets(data)
        self.scat.set_array(np.array(powers))

        return self.scat,

    def update_detections(self, detection_list):
        """
        Thread-safe update of detection list.
        detection_list: list of tuples (azimuth_deg, range_bin, power_db)
        """
        with self.lock:
            self.detections = detection_list

# Test stub
if __name__ == "__main__":
    disp = RadarDisplay()

    def feeder():
        import time
        import random
        while True:
            # Random targets
            dets = []
            for _ in range(5):
                az = random.uniform(0, 360)
                r = random.uniform(10, 90)
                p = random.uniform(10, 30)
                dets.append((az, r, p))
            disp.update_detections(dets)
            time.sleep(1.0)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()

    disp.start()
