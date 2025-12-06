
from PyQt5 import Qt, QtCore
import numpy as np

class RangeDopplerWidget(Qt.QWidget):
    """
    Custom Qt Widget to display the Range-Doppler Map.
    """
    update_signal = QtCore.pyqtSignal(object)

    def __init__(self, fft_len, doppler_len, parent=None):
        super().__init__(parent)
        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.data = np.zeros((doppler_len, fft_len), dtype=np.float32)

        self.update_signal.connect(self.on_update)

        layout = Qt.QVBoxLayout(self)
        self.label = Qt.QLabel()
        self.label.setScaledContents(True) # Allow resizing
        layout.addWidget(self.label)

    def on_update(self, data):
        self.data = data
        # Normalize for display (simple min/max)
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max > d_min:
            norm = (data - d_min) / (d_max - d_min) * 255
        else:
            norm = data * 0

        norm = norm.astype(np.uint8)

        # Map to colormap (grayscale for now)
        # In PyQt, creating an image from numpy array usually involves some shuffling
        # data is (doppler_len, fft_len) -> (Y, X)
        # We want X axis = Doppler, Y axis = Range ?
        # Standard RD Map: X=Doppler, Y=Range.
        # But our matrix is (doppler_len, fft_len).
        # So axis 0 is Doppler, axis 1 is Range.
        # So we should transpose to get (Range, Doppler) -> (Y, X)

        # display_data shape: (fft_len, doppler_len)
        display_data = norm.T

        h, w = display_data.shape
        # Create QImage from data
        # We need to make it contiguous and formatted correctly
        # This is a bit tricky with raw QImage and numpy without extra libs like qimage2ndarray
        # Simple hack: create RGB array

        rgb = np.dstack((display_data, display_data, display_data)).copy()

        qimg = Qt.QImage(rgb.data, w, h, 3*w, Qt.QImage.Format_RGB888)
        self.label.setPixmap(Qt.QPixmap.fromImage(qimg))
