from pydantic.dataclasses import dataclass
from typing import ClassVar
import datetime as dtm

from common.base_class import NameDateClass
from common.chrono.calendar import CalendarID, CalendarContext
from common.models.data_series import DataSeries


FRIDAY_ID = 4
DAYS_WEEK = 7
DAYS_WORK = 5
DAY_SECONDS = 24 * 60 * 60

def get_from_to_adjusted(from_dtm: dtm.datetime, to_dtm: dtm.datetime):
    from_weekday = from_dtm.weekday()
    # if the start date is on a weekend, forward the date to next Monday SOD
    if from_weekday > FRIDAY_ID:
        from_dtm = dtm.datetime.combine(from_dtm, dtm.time.min) + dtm.timedelta(days=DAYS_WEEK - from_weekday)
    # if the end date is on a weekend, rewind the date to the previous Friday EOD
    to_weekday = to_dtm.weekday()
    if to_weekday > FRIDAY_ID:
        to_dtm = dtm.datetime.combine(to_dtm, dtm.time.min) - dtm.timedelta(days=to_weekday - FRIDAY_ID-1)
    return from_dtm, to_dtm

# https://stackoverflow.com/questions/3615375/number-of-days-between-2-dates-excluding-weekends
def get_workdays(from_date: dtm.datetime, to_date: dtm.datetime):
    from_date, to_date = get_from_to_adjusted(from_date, to_date)
    num_days = (to_date - from_date).days
    num_weekdays = (to_date.weekday() - from_date.weekday()) % DAYS_WORK
    time_fraction = (to_date - dtm.datetime.combine(to_date, from_date.time())).seconds / DAY_SECONDS
    return int(num_days / DAYS_WEEK) * DAYS_WORK + num_weekdays + time_fraction

def get_next_slot(from_dtm: dtm.datetime, to_dtm: dtm.datetime):
    slot_start, slot_end = from_dtm, to_dtm # get_from_to_adjusted(from_dtm, to_dtm)
    while slot_start < slot_end:
        slot_end_mid = dtm.datetime.combine(slot_start, dtm.time.min) + dtm.timedelta(days=FRIDAY_ID+1 - slot_start.weekday())
        if slot_end_mid >= slot_end:
            return slot_end, False
        else:
            yield slot_end_mid, False
        slot_start = slot_end_mid + dtm.timedelta(days=2)
        yield slot_start, True


DAYS_YEAR = 365
@dataclass
class EventsCurve(NameDateClass):
    _event_weights: dict[tuple[dtm.datetime, dtm.datetime], float]
    _weekend_weight: float = 0
    _calendar: CalendarID = CalendarID.USD

    _time_series: ClassVar[DataSeries]

    def __post_init__(self):
        self.set_cumulative_series()
    
    def set_cumulative_series(self):
        res = {}
        last_datetime, sum_value = dtm.datetime.combine(self.date, dtm.time.min), 0
        res[last_datetime] = sum_value
        event_weights = self._event_weights
        for holiday in CalendarContext().get_holidays(self._calendar):
            holiday_dtm = dtm.datetime.combine(holiday, dtm.time.min)
            event_weights[(holiday_dtm, holiday_dtm + dtm.timedelta(days=1))] = 0
        for (ev_start, ev_end), weight in sorted(event_weights).items():
            if ev_end <= last_datetime:
                continue
            if ev_start > last_datetime:
                slot_start = last_datetime
                for slot_end, is_weekend in get_next_slot(last_datetime, ev_start):
                    dcf = (slot_end - slot_start).seconds / (DAY_SECONDS * DAYS_YEAR)
                    if is_weekend:
                        sum_value += dcf * self._weekend_weight
                    else:
                        sum_value += dcf
                    res[slot_end] = sum_value
                    slot_start = slot_end
                last_datetime = ev_start
            dcf = (ev_end - last_datetime).seconds / (DAY_SECONDS * DAYS_YEAR)
            sum_value += dcf * weight
            res[ev_end] = sum_value
            last_datetime = ev_end
        self._time_series = DataSeries(res)
    
    def get_effective_time(self, date_time: dtm.datetime):
        next_id = self._time_series.bisect_right(date_time)
        # assert(next_id > 0, f"{date} is before the first available point {self.data.get_first_point()[0]}")
        last_datetime, last_value = self.data.peekitem(next_id-1)
        if last_datetime == date_time:
            return last_value
        last_secs = get_workdays(last_datetime, date_time).seconds
        try:
            next_datetime, next_value = self.data.peekitem(next_id)
        except IndexError:
            return last_value + last_secs / (DAY_SECONDS * DAYS_YEAR)
        slope = (next_value - last_value) / (next_datetime - last_datetime).seconds
        return last_value + slope * last_secs
