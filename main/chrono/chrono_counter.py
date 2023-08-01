from datetime import datetime, timedelta


class ChronoCounter():
    def __init__(self, tolerance = timedelta(0, 500000)):
        self.id_timestamps = dict()
        self.tolerance = tolerance

    def check_ids(self, ids):
        now = datetime.now()
        for id_ in ids:
            if id_ not in self.id_timestamps:
                self.id_timestamps[id_] = {
                    'start': now
                }
            self.id_timestamps[id_]['last'] = now
        for id_ in self.id_timestamps:
            if (now - self.id_timestamps[id_]['last']) > self.tolerance:
                self.id_timestamps.pop(id_)
        return self.id_timestamps






























