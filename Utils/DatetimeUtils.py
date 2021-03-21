import datetime
from pytz import timezone


class DatetimeUtils:

    @staticmethod
    def convert_datetime_object_to_iso_8601_zulu_notation(date_time: datetime.datetime) -> str:
        return date_time.isoformat().replace('+00:00', 'Z')

    @staticmethod
    def get_current_time_as_iso_8601_datetime_string_in_zulu_notation():
        right_now = datetime.datetime.now(tz=timezone('EST'))
        return right_now.isoformat().replace('+00:00', 'Z')
