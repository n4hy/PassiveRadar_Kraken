
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
    """Mock replacement for gr.top_block in test environments.

    Technique: stub methods that do nothing to satisfy interface requirements.
    """
    def __init__(self, name="MockTopBlock"):
        """Initialize mock top block with optional name."""
        pass
    def connect(self, *args):
        """Stub for connecting blocks in a flowgraph."""
        pass
    def start(self):
        """Stub for starting flowgraph execution."""
        pass
    def stop(self):
        """Stub for stopping flowgraph execution."""
        pass
    def wait(self):
        """Stub for waiting on flowgraph completion."""
        pass
    def show(self):
        """Stub for showing the flowgraph GUI."""
        pass

class MockBasicBlock:
    """Mock replacement for gr.basic_block in test environments.

    Technique: stub methods to satisfy GNU Radio block interface.
    """
    def __init__(self, name="MockBasicBlock", in_sig=None, out_sig=None):
        """Initialize mock basic block with name and I/O signatures."""
        pass
    def consume(self, port, count):
        """Stub for consuming input items on a given port."""
        pass

class MockHierBlock2:
    """Mock replacement for gr.hier_block2 in test environments.

    Technique: stub methods to satisfy hierarchical block interface.
    """
    def __init__(self, name, in_sig, out_sig):
        """Initialize mock hierarchical block with name and I/O signatures."""
        pass
    def connect(self, *args):
        """Stub for connecting internal blocks."""
        pass

gr.top_block = MockTopBlock
gr.basic_block = MockBasicBlock
gr.hier_block2 = MockHierBlock2
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
filter = filter_mod  # Expose as 'filter' so 'from gnuradio import filter' works when gnuradio is mocked by this module

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

# Create a mock module for soapy (SDRplay RSPdx via SoapySDR)
soapy = MagicMock()
sys.modules["gnuradio.soapy"] = soapy
gnuradio.soapy = soapy

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
    """Mock replacement for QtCore.QObject in test environments.

    Technique: stub constructor to satisfy Qt object inheritance.
    """
    def __init__(self, parent=None):
        """Initialize mock QObject with optional parent."""
        pass

class MockpyqtSignal:
    """Mock replacement for QtCore.pyqtSignal in test environments.

    Technique: stub signal emission and connection methods.
    """
    def __init__(self, *args):
        """Initialize mock signal with type arguments."""
        pass
    def emit(self, *args):
        """Stub for emitting a signal."""
        pass
    def connect(self, func):
        """Stub for connecting a slot to this signal."""
        pass

QtCore.QObject = MockQObject
QtCore.pyqtSignal = MockpyqtSignal

class MockQWidget:
    """Mock replacement for Qt.QWidget in test environments.

    Technique: stub all widget methods to satisfy Qt widget interface.
    """
    def __init__(self, parent=None):
        """Initialize mock widget with optional parent."""
        pass
    def setWindowTitle(self, title):
        """Stub for setting window title."""
        pass
    def closeEvent(self, event):
        """Stub for handling close events."""
        pass
    def show(self):
        """Stub for showing the widget."""
        pass
    def widget(self):
        """Stub that returns a new MockQWidget instance."""
        return MockQWidget()
    def update(self):
        """Stub for triggering widget repaint."""
        pass
    def setScaledContents(self, val):
        """Stub for setting scaled contents mode."""
        pass
    def setPixmap(self, val):
        """Stub for setting a pixmap on the widget."""
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
