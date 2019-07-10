from datetime import date, datetime, time as dt_time, timedelta

import numpy as np
import pytest

from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    Timestamp,
    conversion,
    timezones,
)
from pandas._libs.tslibs.frequencies import (
    INVALID_FREQ_ERR_MSG,
    get_freq_code,
    get_freq_str,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import ApplyTypeError
import pandas.compat as compat
from pandas.compat.numpy import np_datetime64_compat

from pandas.core.indexes.datetimes import DatetimeIndex, _to_M8, date_range
from pandas.core.series import Series
import pandas.util.testing as tm

from pandas.io.pickle import read_pickle
from pandas.tseries.frequencies import _offset_map, get_offset
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas.tseries.offsets as offsets
from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessHour,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CDay,
    CustomBusinessHour,
    #DateOffset,
    Day,
    Easter,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    MonthEnd,
    Nano,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Tick,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)

from .common import assert_offset_equal, assert_onOffset


class WeekDay:
    # TODO: Remove: This is not used outside of tests
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6


####
# Misc function tests
####


def test_to_M8():
    valb = datetime(2007, 10, 1)
    valu = _to_M8(valb)
    assert isinstance(valu, np.datetime64)


#####
# DateOffset Tests
#####


class Base:
    _offset = None
    d = Timestamp(datetime(2008, 1, 2))

    timezones = [
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Tokyo",
        "dateutil/US/Pacific",
    ]

    def _get_offset(self, klass, value=1, normalize=False):
        # create instance from offset class
        if klass is FY5253:
            klass = klass(
                n=value,
                startingMonth=1,
                weekday=1,
                variation="last",
                normalize=normalize,
            )
        elif klass is FY5253Quarter:
            klass = klass(
                n=value,
                startingMonth=1,
                weekday=1,
                qtr_with_extra_week=1,
                variation="last",
                normalize=normalize,
            )
        elif klass is LastWeekOfMonth:
            klass = klass(n=value, weekday=5, normalize=normalize)
        elif klass is WeekOfMonth:
            klass = klass(n=value, week=1, weekday=5, normalize=normalize)
        elif klass is Week:
            klass = klass(n=value, weekday=5, normalize=normalize)
        elif klass is DateOffset:
            klass = klass(days=value, normalize=normalize)
        else:
            try:
                klass = klass(value, normalize=normalize)
            except Exception:
                klass = klass(normalize=normalize)
        return klass

    def test_apply_out_of_range(self, tz_naive_fixture):
        tz = tz_naive_fixture
        if self._offset is None:
            return

        # try to create an out-of-bounds result timestamp; if we can't create
        # the offset skip
        try:
            if self._offset in (BusinessHour, CustomBusinessHour):
                # Using 10000 in BusinessHour fails in tz check because of DST
                # difference
                offset = self._get_offset(self._offset, value=100000)
            else:
                offset = self._get_offset(self._offset, value=10000)

            result = Timestamp("20080101") + offset
            assert isinstance(result, datetime)
            assert result.tzinfo is None

            # Check tz is preserved
            t = Timestamp("20080101", tz=tz)
            result = t + offset
            assert isinstance(result, datetime)
            assert t.tzinfo == result.tzinfo

        except OutOfBoundsDatetime:
            pass
        except (ValueError, KeyError):
            # we are creating an invalid offset
            # so ignore
            pass

    def test_offsets_compare_equal(self):
        # root cause of GH#456: __ne__ was not implemented
        if self._offset is None:
            return
        offset1 = self._offset()
        offset2 = self._offset()
        assert not offset1 != offset2
        assert offset1 == offset2

    def test_rsub(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        assert self.d - self.offset2 == (-self.offset2).apply(self.d)

    def test_radd(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        assert self.d + self.offset2 == self.offset2 + self.d

    def test_sub(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        off = self.offset2
        msg = "Cannot subtract datetime from offset"
        with pytest.raises(TypeError, match=msg):
            off - self.d

        assert 2 * off - off == off
        assert self.d - self.offset2 == self.d + self._offset(-2)
        assert self.d - self.offset2 == self.d - (2 * off - off)

    def testMult1(self):
        if self._offset is None or not hasattr(self, "offset1"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset1 attr
            return
        assert self.d + 10 * self.offset1 == self.d + self._offset(10)
        assert self.d + 5 * self.offset1 == self.d + self._offset(5)

    def testMult2(self):
        if self._offset is None:
            return
        assert self.d + (-5 * self._offset(-10)) == self.d + self._offset(50)
        assert self.d + (-3 * self._offset(-2)) == self.d + self._offset(6)

    def test_compare_str(self):
        # GH#23524
        # comparing to strings that cannot be cast to DateOffsets should
        #  not raise for __eq__ or __ne__
        if self._offset is None:
            return
        off = self._get_offset(self._offset)

        assert not off == "infer"
        assert off != "foo"
        # Note: inequalities are only implemented for Tick subclasses;
        #  tests for this are in test_ticks


class TestWeek(Base):
    _offset = Week
    d = Timestamp(datetime(2008, 1, 2))
    offset1 = _offset()
    offset2 = _offset(2)

    def test_repr(self):
        assert repr(Week(weekday=0)) == "<Week: weekday=0>"
        assert repr(Week(n=-1, weekday=0)) == "<-1 * Week: weekday=0>"
        assert repr(Week(n=-2, weekday=0)) == "<-2 * Weeks: weekday=0>"

    def test_corner(self):
        with pytest.raises(ValueError):
            Week(weekday=7)

        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=-1)

    def test_isAnchored(self):
        assert Week(weekday=0).isAnchored()
        assert not Week().isAnchored()
        assert not Week(2, weekday=2).isAnchored()
        assert not Week(2).isAnchored()

    offset_cases = []
    # not business week
    offset_cases.append(
        (
            Week(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 8),
                datetime(2008, 1, 4): datetime(2008, 1, 11),
                datetime(2008, 1, 5): datetime(2008, 1, 12),
                datetime(2008, 1, 6): datetime(2008, 1, 13),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # Mon
    offset_cases.append(
        (
            Week(weekday=0),
            {
                datetime(2007, 12, 31): datetime(2008, 1, 7),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(0, weekday=0),
            {
                datetime(2007, 12, 31): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(-2, weekday=1),
            {
                datetime(2010, 4, 6): datetime(2010, 3, 23),
                datetime(2010, 4, 8): datetime(2010, 3, 30),
                datetime(2010, 4, 5): datetime(2010, 3, 23),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("weekday", range(7))
    def test_onOffset(self, weekday):
        offset = Week(weekday=weekday)

        for day in range(1, 8):
            date = datetime(2008, 1, day)

            if day % 7 == weekday:
                expected = True
            else:
                expected = False
        assert_onOffset(offset, date, expected)


class TestWeekOfMonth(Base):
    _offset = WeekOfMonth
    offset1 = _offset()
    offset2 = _offset(2)

    def test_constructor(self):
        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=4, weekday=0)

        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=-1, weekday=0)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-7)

    def test_repr(self):
        assert (
            repr(WeekOfMonth(weekday=1, week=2)) == "<WeekOfMonth: week=2, weekday=1>"
        )

    def test_offset(self):
        date1 = datetime(2011, 1, 4)  # 1st Tuesday of Month
        date2 = datetime(2011, 1, 11)  # 2nd Tuesday of Month
        date3 = datetime(2011, 1, 18)  # 3rd Tuesday of Month
        date4 = datetime(2011, 1, 25)  # 4th Tuesday of Month

        # see for loop for structure
        test_cases = [
            (-2, 2, 1, date1, datetime(2010, 11, 16)),
            (-2, 2, 1, date2, datetime(2010, 11, 16)),
            (-2, 2, 1, date3, datetime(2010, 11, 16)),
            (-2, 2, 1, date4, datetime(2010, 12, 21)),
            (-1, 2, 1, date1, datetime(2010, 12, 21)),
            (-1, 2, 1, date2, datetime(2010, 12, 21)),
            (-1, 2, 1, date3, datetime(2010, 12, 21)),
            (-1, 2, 1, date4, datetime(2011, 1, 18)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 0, 1, date2, datetime(2011, 2, 1)),
            (0, 0, 1, date3, datetime(2011, 2, 1)),
            (0, 0, 1, date4, datetime(2011, 2, 1)),
            (0, 1, 1, date1, datetime(2011, 1, 11)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 1, 1, date3, datetime(2011, 2, 8)),
            (0, 1, 1, date4, datetime(2011, 2, 8)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 2, 1, date3, datetime(2011, 1, 18)),
            (0, 3, 1, date4, datetime(2011, 1, 25)),
            (1, 0, 0, date1, datetime(2011, 2, 7)),
            (1, 0, 0, date2, datetime(2011, 2, 7)),
            (1, 0, 0, date3, datetime(2011, 2, 7)),
            (1, 0, 0, date4, datetime(2011, 2, 7)),
            (1, 0, 1, date1, datetime(2011, 2, 1)),
            (1, 0, 1, date2, datetime(2011, 2, 1)),
            (1, 0, 1, date3, datetime(2011, 2, 1)),
            (1, 0, 1, date4, datetime(2011, 2, 1)),
            (1, 0, 2, date1, datetime(2011, 1, 5)),
            (1, 0, 2, date2, datetime(2011, 2, 2)),
            (1, 0, 2, date3, datetime(2011, 2, 2)),
            (1, 0, 2, date4, datetime(2011, 2, 2)),
            (1, 2, 1, date1, datetime(2011, 1, 18)),
            (1, 2, 1, date2, datetime(2011, 1, 18)),
            (1, 2, 1, date3, datetime(2011, 2, 15)),
            (1, 2, 1, date4, datetime(2011, 2, 15)),
            (2, 2, 1, date1, datetime(2011, 2, 15)),
            (2, 2, 1, date2, datetime(2011, 2, 15)),
            (2, 2, 1, date3, datetime(2011, 3, 15)),
            (2, 2, 1, date4, datetime(2011, 3, 15)),
        ]

        for n, week, weekday, dt, expected in test_cases:
            offset = WeekOfMonth(n, week=week, weekday=weekday)
            assert_offset_equal(offset, dt, expected)

        # try subtracting
        result = datetime(2011, 2, 1) - WeekOfMonth(week=1, weekday=2)
        assert result == datetime(2011, 1, 12)

        result = datetime(2011, 2, 3) - WeekOfMonth(week=0, weekday=2)
        assert result == datetime(2011, 2, 2)

    on_offset_cases = [
        (0, 0, datetime(2011, 2, 7), True),
        (0, 0, datetime(2011, 2, 6), False),
        (0, 0, datetime(2011, 2, 14), False),
        (1, 0, datetime(2011, 2, 14), True),
        (0, 1, datetime(2011, 2, 1), True),
        (0, 1, datetime(2011, 2, 8), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_onOffset(self, case):
        week, weekday, dt, expected = case
        offset = WeekOfMonth(week=week, weekday=weekday)
        assert offset.onOffset(dt) == expected


class TestLastWeekOfMonth(Base):
    _offset = LastWeekOfMonth
    offset1 = _offset()
    offset2 = _offset(2)

    def test_constructor(self):
        with pytest.raises(ValueError, match="^N cannot be 0"):
            LastWeekOfMonth(n=0, weekday=1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=7)

    def test_offset(self):
        # Saturday
        last_sat = datetime(2013, 8, 31)
        next_sat = datetime(2013, 9, 28)
        offset_sat = LastWeekOfMonth(n=1, weekday=5)

        one_day_before = last_sat + timedelta(days=-1)
        assert one_day_before + offset_sat == last_sat

        one_day_after = last_sat + timedelta(days=+1)
        assert one_day_after + offset_sat == next_sat

        # Test On that day
        assert last_sat + offset_sat == next_sat

        # Thursday

        offset_thur = LastWeekOfMonth(n=1, weekday=3)
        last_thurs = datetime(2013, 1, 31)
        next_thurs = datetime(2013, 2, 28)

        one_day_before = last_thurs + timedelta(days=-1)
        assert one_day_before + offset_thur == last_thurs

        one_day_after = last_thurs + timedelta(days=+1)
        assert one_day_after + offset_thur == next_thurs

        # Test on that day
        assert last_thurs + offset_thur == next_thurs

        three_before = last_thurs + timedelta(days=-3)
        assert three_before + offset_thur == last_thurs

        two_after = last_thurs + timedelta(days=+2)
        assert two_after + offset_thur == next_thurs

        offset_sunday = LastWeekOfMonth(n=1, weekday=WeekDay.SUN)
        assert datetime(2013, 7, 31) + offset_sunday == datetime(2013, 8, 25)

    on_offset_cases = [
        (WeekDay.SUN, datetime(2013, 1, 27), True),
        (WeekDay.SAT, datetime(2013, 3, 30), True),
        (WeekDay.MON, datetime(2013, 2, 18), False),  # Not the last Mon
        (WeekDay.SUN, datetime(2013, 2, 25), False),  # Not a SUN
        (WeekDay.MON, datetime(2013, 2, 25), True),
        (WeekDay.SAT, datetime(2013, 11, 30), True),
        (WeekDay.SAT, datetime(2006, 8, 26), True),
        (WeekDay.SAT, datetime(2007, 8, 25), True),
        (WeekDay.SAT, datetime(2008, 8, 30), True),
        (WeekDay.SAT, datetime(2009, 8, 29), True),
        (WeekDay.SAT, datetime(2010, 8, 28), True),
        (WeekDay.SAT, datetime(2011, 8, 27), True),
        (WeekDay.SAT, datetime(2019, 8, 31), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_onOffset(self, case):
        weekday, dt, expected = case
        offset = LastWeekOfMonth(weekday=weekday)
        assert offset.onOffset(dt) == expected


def test_Easter():
    assert_offset_equal(Easter(), datetime(2010, 1, 1), datetime(2010, 4, 4))
    assert_offset_equal(Easter(), datetime(2010, 4, 5), datetime(2011, 4, 24))
    assert_offset_equal(Easter(2), datetime(2010, 1, 1), datetime(2011, 4, 24))

    assert_offset_equal(Easter(), datetime(2010, 4, 4), datetime(2011, 4, 24))
    assert_offset_equal(Easter(2), datetime(2010, 4, 4), datetime(2012, 4, 8))

    assert_offset_equal(-Easter(), datetime(2011, 1, 1), datetime(2010, 4, 4))
    assert_offset_equal(-Easter(), datetime(2010, 4, 5), datetime(2010, 4, 4))
    assert_offset_equal(-Easter(2), datetime(2011, 1, 1), datetime(2009, 4, 12))

    assert_offset_equal(-Easter(), datetime(2010, 4, 4), datetime(2009, 4, 12))
    assert_offset_equal(-Easter(2), datetime(2010, 4, 4), datetime(2008, 3, 23))


class TestOffsetNames:
    def test_get_offset_name(self):
        assert BDay().freqstr == "B"
        assert BDay(2).freqstr == "2B"
        assert BMonthEnd().freqstr == "BM"
        assert Week(weekday=0).freqstr == "W-MON"
        assert Week(weekday=1).freqstr == "W-TUE"
        assert Week(weekday=2).freqstr == "W-WED"
        assert Week(weekday=3).freqstr == "W-THU"
        assert Week(weekday=4).freqstr == "W-FRI"

        assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"


def test_get_offset_legacy():
    pairs = [("w@Sat", Week(weekday=5))]
    for name, expected in pairs:
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            get_offset(name)


class TestOffsetAliases:
    def setup_method(self, method):
        _offset_map.clear()

    def test_alias_equality(self):
        for k, v in _offset_map.items():
            if v is None:
                continue
            assert k == v.copy()

    def test_rule_code(self):
        lst = ["M", "MS", "BM", "BMS", "D", "B", "H", "T", "S", "L", "U"]
        for k in lst:
            assert k == get_offset(k).rule_code
            # should be cached - this is kind of an internals test...
            assert k in _offset_map
            assert k == (get_offset(k) * 3).rule_code

        suffix_lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        base = "W"
        for v in suffix_lst:
            alias = "-".join([base, v])
            assert alias == get_offset(alias).rule_code
            assert alias == (get_offset(alias) * 5).rule_code

        suffix_lst = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
        base_lst = ["A", "AS", "BA", "BAS", "Q", "QS", "BQ", "BQS"]
        for base in base_lst:
            for v in suffix_lst:
                alias = "-".join([base, v])
                assert alias == get_offset(alias).rule_code
                assert alias == (get_offset(alias) * 5).rule_code

        lst = ["M", "D", "B", "H", "T", "S", "L", "U"]
        for k in lst:
            code, stride = get_freq_code("3" + k)
            assert isinstance(code, int)
            assert stride == 3
            assert k == get_freq_str(code)


def test_dateoffset_misc():
    oset = offsets.DateOffset(months=2, days=4)
    # it works
    oset.freqstr

    assert not offsets.DateOffset(months=2) == 2


def test_freq_offsets():
    off = BDay(1, offset=timedelta(0, 1800))
    assert off.freqstr == "B+30Min"

    off = BDay(1, offset=timedelta(0, -1800))
    assert off.freqstr == "B-30Min"


class TestReprNames:
    def test_str_for_named_is_name(self):
        # look at all the amazing combinations!
        month_prefixes = ["A", "AS", "BA", "BAS", "Q", "BQ", "BQS", "QS"]
        names = [
            prefix + "-" + month
            for prefix in month_prefixes
            for month in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ]
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        names += ["W-" + day for day in days]
        names += ["WOM-" + week + day for week in ("1", "2", "3", "4") for day in days]
        _offset_map.clear()
        for name in names:
            offset = get_offset(name)
            assert offset.freqstr == name

# ---------------------------------------------------------------------


def test_valid_default_arguments(offset_types):
    # GH#19142 check that the calling the constructors without passing
    # any keyword arguments produce valid offsets
    cls = offset_types
    cls()


@pytest.mark.parametrize("kwd", sorted(list(liboffsets.relativedelta_kwds)))
def test_valid_month_attributes(kwd, month_classes):
    # GH#18226
    cls = month_classes
    # check that we cannot create e.g. MonthEnd(weeks=3)
    with pytest.raises(TypeError):
        cls(**{kwd: 3})


@pytest.mark.parametrize("kwd", sorted(list(liboffsets.relativedelta_kwds)))
def test_valid_tick_attributes(kwd, tick_classes):
    # GH#18226
    cls = tick_classes
    # check that we cannot create e.g. Hour(weeks=3)
    with pytest.raises(TypeError):
        cls(**{kwd: 3})


def test_require_integers(offset_types):
    cls = offset_types
    with pytest.raises(ValueError):
        cls(n=1.5)


def test_tick_normalize_raises(tick_classes):
    # check that trying to create a Tick object with normalize=True raises
    # GH#21427
    cls = tick_classes
    with pytest.raises(ValueError):
        cls(n=3, normalize=True)


def test_weeks_onoffset():
    # GH#18510 Week with weekday = None, normalize = False should always
    # be onOffset
    offset = Week(n=2, weekday=None)
    ts = Timestamp("1862-01-13 09:03:34.873477378+0210", tz="Africa/Lusaka")
    fast = offset.onOffset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow

    # negative n
    offset = Week(n=2, weekday=None)
    ts = Timestamp("1856-10-24 16:18:36.556360110-0717", tz="Pacific/Easter")
    fast = offset.onOffset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow


def test_weekofmonth_onoffset():
    # GH#18864
    # Make sure that nanoseconds don't trip up onOffset (and with it apply)
    offset = WeekOfMonth(n=2, week=2, weekday=0)
    ts = Timestamp("1916-05-15 01:14:49.583410462+0422", tz="Asia/Qyzylorda")
    fast = offset.onOffset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow

    # negative n
    offset = WeekOfMonth(n=-3, week=1, weekday=0)
    ts = Timestamp("1980-12-08 03:38:52.878321185+0500", tz="Asia/Oral")
    fast = offset.onOffset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow


def test_last_week_of_month_on_offset():
    # GH#19036, GH#18977 _adjust_dst was incorrect for LastWeekOfMonth
    offset = LastWeekOfMonth(n=4, weekday=6)
    ts = Timestamp("1917-05-27 20:55:27.084284178+0200", tz="Europe/Warsaw")
    slow = (ts + offset) - offset == ts
    fast = offset.onOffset(ts)
    assert fast == slow

    # negative n
    offset = LastWeekOfMonth(n=-4, weekday=5)
    ts = Timestamp("2005-08-27 05:01:42.799392561-0500", tz="America/Rainy_River")
    slow = (ts + offset) - offset == ts
    fast = offset.onOffset(ts)
    assert fast == slow
