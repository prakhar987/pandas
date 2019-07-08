from pandas._libs.tslibs import Timestamp
from pandas.tseries.frequencies import _offset_map

from datetime import datetime

from pandas.tseries.offsets import DateOffset
from .test_business_offsets import Base


class TestDateOffset(Base):
    def setup_method(self, method):
        self.d = Timestamp(datetime(2008, 1, 2))
        _offset_map.clear()

    def test_repr(self):
        repr(DateOffset())
        repr(DateOffset(2))
        repr(2 * DateOffset())
        repr(2 * DateOffset(months=2))

    def test_mul(self):
        assert DateOffset(2) == 2 * DateOffset(1)
        assert DateOffset(2) == DateOffset(1) * 2

    def test_constructor(self):

        assert (self.d + DateOffset(months=2)) == datetime(2008, 3, 2)
        assert (self.d - DateOffset(months=2)) == datetime(2007, 11, 2)

        assert (self.d + DateOffset(2)) == datetime(2008, 1, 4)

        assert not DateOffset(2).isAnchored()
        assert DateOffset(1).isAnchored()

        d = datetime(2008, 1, 31)
        assert (d + DateOffset(months=1)) == datetime(2008, 2, 29)

    def test_copy(self):
        assert DateOffset(months=2).copy() == DateOffset(months=2)

    def test_eq(self):
        offset1 = DateOffset(days=1)
        offset2 = DateOffset(days=365)

        assert offset1 != offset2