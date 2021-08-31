class TimeInterval:
    def __init__(self, freq):
        self.freq = int(freq.split(' ')[0])
        self.unit = freq.split()[1].lower()
        self.last_logged = 0

    def get_value(self, counters):
        if self.unit in counters:
            value = counters[self.unit]
        elif self.unit.endswith('s') and self.unit[:-1] in counters:
            value = counters[self.unit[:-1]]
        else:
            raise Exception(
                'Invalid unit %s, the valid options are %s' % (
                    self.unit,
                    ', '.join([c+'(s)' for c in counters]))
            )
        return value

    def is_time(self, counters):
        return (self.get_value(counters) - self.last_logged) >= self.freq

    def update(self, counters):
        self.last_logged = self.get_value(counters)

    def get_progress(self, counters):
        return self.get_value(counters)-self.last_logged

    def percentage(self, counters):
        return int((self.get_value(counters)-self.last_logged)/self.freq * 100)

