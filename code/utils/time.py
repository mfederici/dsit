class TimeInterval:
    def __init__(self, freq):
        self.freq = int(freq.split(' ')[0])
        self.unit = freq.split()[1].lower()
        self.last_logged = 0

    def is_time(self, model):
        if hasattr(model, self.unit):
            unit = self.unit
        elif self.unit.endswith('s') and hasattr(model, self.unit[:-1]):
            unit = self.unit[:-1]
        if not hasattr(model, self.unit):
            raise Exception('Invalid unit %s' % (self.unit))

        curr_value = getattr(model, unit)

        return (curr_value-self.last_logged) >= self.freq

    def update(self, model):
        self.last_logged = getattr(model, self.unit)

    def get_progress(self, model):
        curr_value = getattr(model, self.unit)
        return curr_value-self.last_logged

    def percentage(self, model):
        curr_value = getattr(model, self.unit)
        return int((curr_value-self.last_logged)/self.freq * 100)

