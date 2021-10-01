SCALAR_ENTRY = 'scalar'
SCALARS_ENTRY = 'scalars'
IMAGE_ENTRY = 'image'
PLOT_ENTRY = 'plot'

LOG_ENTRY_TYPES = []


class LogEntry:
    def __init__(self, value, data_type):
        self.value = value
        self.data_type = data_type

    def __repr__(self):
        return 'LogEntry(\n   %s\n)' % self.value.__repr__().replace('\n', '\n   ')



