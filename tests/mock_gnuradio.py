
import sys
from unittest.mock import MagicMock

# Create a mock module structure
gnuradio = MagicMock()
sys.modules["gnuradio"] = gnuradio

# Mock gr
gr = MagicMock()
sys.modules["gnuradio.gr"] = gr
gnuradio.gr = gr

class MockTopBlock:
    def __init__(self, name="MockTopBlock"):
        pass
    def connect(self, *args):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def wait(self):
        pass
    def show(self):
        pass

class MockBasicBlock:
    def __init__(self, name="MockBasicBlock", in_sig=None, out_sig=None):
        pass
    def consume(self, port, count):
        pass

gr.top_block = MockTopBlock
gr.basic_block = MockBasicBlock
gr.sizeof_gr_complex = 8
gr.sizeof_float = 4

# Mock blocks
blocks = MagicMock()
sys.modules["gnuradio.blocks"] = blocks
gnuradio.blocks = blocks

# Mock filter
filter_mod = MagicMock()
sys.modules["gnuradio.filter"] = filter_mod
gnuradio.filter = filter_mod

# Mock analog
analog = MagicMock()
sys.modules["gnuradio.analog"] = analog
gnuradio.analog = analog

# Mock fft
fft = MagicMock()
sys.modules["gnuradio.fft"] = fft
gnuradio.fft = fft

# Mock qtgui
qtgui = MagicMock()
sys.modules["gnuradio.qtgui"] = qtgui
gnuradio.qtgui = qtgui

# Create a mock module for osmosdr
osmosdr = MagicMock()
sys.modules["osmosdr"] = osmosdr

# Create a mock module for PyQt5
PyQt5 = MagicMock()
sys.modules["PyQt5"] = PyQt5
Qt = MagicMock()
PyQt5.Qt = Qt
sys.modules["PyQt5.Qt"] = Qt
QtCore = MagicMock()
PyQt5.QtCore = QtCore
sys.modules["PyQt5.QtCore"] = QtCore

class MockQObject:
    def __init__(self, parent=None):
        pass

class MockpyqtSignal:
    def __init__(self, *args):
        pass
    def emit(self, *args):
        pass
    def connect(self, func):
        pass

QtCore.QObject = MockQObject
QtCore.pyqtSignal = MockpyqtSignal

class MockQWidget:
    def __init__(self, parent=None):
        pass
    def setWindowTitle(self, title):
        pass
    def closeEvent(self, event):
        pass
    def show(self):
        pass
    def widget(self):
        return MockQWidget()
    def update(self):
        pass
    def setScaledContents(self, val):
        pass
    def setPixmap(self, val):
        pass

Qt.QWidget = MockQWidget
Qt.QVBoxLayout = MagicMock()
Qt.QHBoxLayout = MagicMock()
Qt.QScrollArea = MagicMock()
Qt.QFrame = MagicMock()
Qt.QApplication = MagicMock()
Qt.QLabel = MagicMock()
Qt.QImage = MagicMock()
Qt.QPixmap = MagicMock()

# Mock firdes
firdes = MagicMock()
filter_mod.firdes = firdes
firdes.low_pass.return_value = [1.0]
firdes.WIN_HAMMING = 1
firdes.WIN_BLACKMAN_hARRIS = 2

# Mock sip
sip = MagicMock()
sys.modules["sip"] = sip
