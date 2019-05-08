###############################################################################
#   volumina: volume slicing and editing library
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
# 		   http://ilastik.org/license/
###############################################################################
import pytest

import unittest as ut
import os
from abc import ABCMeta, abstractmethod
from numpy.testing import assert_array_almost_equal
from unittest import mock
import volumina._testing
from volumina.pixelpipeline.datasources import ArraySource, RelabelingArraySource, RequestABC
import numpy as np
from volumina.slicingtools import sl, slicing2shape
from future.utils import with_metaclass

try:
    import lazyflow

    has_lazyflow = True
except ImportError:
    has_lazyflow = False

if has_lazyflow:
    from lazyflow.graph import Graph
    from volumina.pixelpipeline._testing import OpDataProvider
    from volumina.pixelpipeline.datasources import LazyflowSource, LazyflowSinkSource


class GenericArraySourceTest(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def setUp(self):
        self.slicing = (slice(0, 1), slice(10, 20), slice(20, 25), slice(0, 1), slice(0, 1))
        self.source = None

    def testRequestWait(self):
        slicing = self.slicing
        requested = self.source.request(slicing).wait()
        self.assertTrue(np.all(requested == self.raw[slicing]))

    def testSetDirty(self):
        self.signal_emitted = False

        def slot(sl):
            self.signal_emitted = True
            self.assertTrue(sl == self.slicing)

        self.source.isDirty.connect(slot)
        self.source.setDirty(self.slicing)
        self.source.isDirty.disconnect(slot)

        self.assertTrue(self.signal_emitted)

        del self.signal_emitted
        del self.slicing

    def testComparison(self):
        assert self.samesource == self.source
        assert self.othersource != self.source


class ArraySourceTest(ut.TestCase, GenericArraySourceTest):
    def setUp(self):
        GenericArraySourceTest.setUp(self)
        self.lena = np.load(os.path.join(volumina._testing.__path__[0], "lena.npy"))
        self.raw = np.zeros((1, 512, 512, 1, 1))
        self.raw[0, :, :, 0, 0] = self.lena
        self.source = ArraySource(self.raw)

        self.samesource = ArraySource(self.raw)
        self.othersource = ArraySource(np.array(self.raw))


class RelabelingArraySourceTest(ut.TestCase, GenericArraySourceTest):
    def setUp(self):
        GenericArraySourceTest.setUp(self)
        a = np.zeros((5, 1, 1, 1, 1), dtype=np.uint32)
        # the data contained in a ranges from [1,5]
        a[:, 0, 0, 0, 0] = np.arange(0, 5)
        self.source = RelabelingArraySource(a)

        # we apply the relabeling i -> i+1
        relabeling = np.arange(1, a.max() + 2, dtype=np.uint32)
        self.source.setRelabeling(relabeling)

        self.samesource = RelabelingArraySource(a)
        self.othersource = RelabelingArraySource(np.array(a))

    def testRequestWait(self):
        slicing = (slice(0, 5), slice(None), slice(None), slice(None), slice(None))
        requested = self.source.request(slicing).wait()
        assert requested.ndim == 5
        self.assertTrue(np.all(requested.flatten() == np.arange(1, 6, dtype=np.uint32)))

    def testSetDirty(self):
        self.signal_emitted = False
        self.slicing = (slice(0, 5), slice(None), slice(None), slice(None), slice(None))

        def slot(sl):
            self.signal_emitted = True
            self.assertTrue(sl == self.slicing)

        self.source.isDirty.connect(slot)
        self.source.setDirty(self.slicing)
        self.source.isDirty.disconnect(slot)

        self.assertTrue(self.signal_emitted)

        del self.signal_emitted
        del self.slicing


class CachedSource:
    """
    Decorator data source provides cachable requests to underlying sources
    """
    class _Cache:
        def __init__(self):
            self._cache = []

        def put(self, key, value):
            self._cache.append((key, value))

        def __iter__(self):
            return iter(self._cache)

    class _CacheRequest:
        def __init__(self, cache, slicing, request_func):
            self._slicing = slicing
            self._cache = cache
            self._request_func = request_func

        def wait(self):
            for key, entry in self._cache:
                if covers(key, self._slicing):
                    subslice = change_origin(key, self._slicing)
                    return entry[subslice]

            result = self._request_func(self._slicing).wait()
            self._cache.put(self._slicing, result)
            return result

    def __init__(self, orig):
        self._orig_src = orig
        self._cache = self._Cache()

    def dtype(self):
        return self._orig_src.dtype()

    def request(self, slicing):
        return self._CacheRequest(self._cache, slicing, self._orig_src.request)


def covers(slicing, other):
    assert len(slicing) == len(other)

    for slice_, other_slice in zip(slicing, other):
        start_covered = (
            slice_.start is None
            or (other_slice.start is not None and slice_.start <= other_slice.start)
        )

        stop_covered = (
            slice_.stop is None
            or (other_slice.stop is not None and slice_.stop >= other_slice.stop)
        )

        if not (start_covered and stop_covered):
            return False

    return True


def change_origin(slicing, other):
    # precondition: covers(slicing, other)
    result = []
    for slice_, other_slice in zip(slicing, other):
        start, stop = None, None
        origin_start = slice_.start or 0

        if other_slice.start is not None:
            start = other_slice.start - origin_start

        if other_slice.stop is not None:
            stop = other_slice.stop - origin_start

        result.append(slice(start, stop))

    # postcondition: arr[slicing][result] == arr[other]
    return tuple(result)


class TestCachedImageSource:
    @pytest.fixture
    def data(self):
        data = np.array(range(512 * 512))
        data.shape = (1, 1, 1, 512, 512)
        return data

    @pytest.fixture
    def orig_src(self, data):
        src = ArraySource(data)
        mocked = ut.mock.Mock(wraps=src)
        return mocked

    @pytest.fixture
    def src(self, orig_src):
        return CachedSource(orig_src)

    def test_request_slicing(self, src, orig_src):
        slicing = sl[:, :, :, :, :]
        src.request(slicing).wait()
        orig_src.request.assert_called_once_with(slicing)

    def test_consecutive_requests_with_same_shape_are_cached(self, src, orig_src, data):
        slicing = sl[:100, :100, :100, :100, :100]
        expected_res = data[0:1, 0:1, 0:1, 0:100, 0:100]

        assert_array_almost_equal(src.request(slicing).wait(), expected_res)
        assert_array_almost_equal(src.request(slicing).wait(), expected_res)

        orig_src.request.assert_called_once_with(slicing)

    def test_request_of_subslicing(self, src, orig_src, data):
        slicing = sl[:100, :100, :100, :100, :100]
        smaller_slicing = sl[:1, :1, :1, :50, :50]

        expected_res = data[0:1, 0:1, 0:1, 0:100, 0:100]
        expected_res_smaller = data[0:1, 0:1, 0:1, 0:50, 0:50]

        assert_array_almost_equal(src.request(slicing).wait(), expected_res)
        assert_array_almost_equal(src.request(smaller_slicing).wait(), expected_res_smaller)

        orig_src.request.assert_called_once_with(slicing)

    def test_has_the_same_dtype_as_original(self, orig_src, src):
        assert orig_src.dtype() == src.dtype()

    @pytest.mark.parametrize('slice1,slice2,result', [
        [sl[0:100, 0:100], sl[0:40, 0:40], True],
        [sl[0:40, 0:40], sl[0:100, 0:100], False],
        [sl[1:40, 1:40], sl[0:40, 0:40], False],
        [sl[1:40, 1:40], sl[5:20, 38:45], False],
        [sl[1:40, 1:40], sl[0:5, 5:20], False],
        [sl[1:40, 1:40], sl[1:39, 1:40], True],
        [sl[:40, :40], sl[:39, :40], True],
        [sl[:40, :40], sl[0:39, 0:40], True],
        [sl[:40, :40], sl[:41, :40], False],
    ])
    def test_covers(self, slice1, slice2, result):
        assert covers(slice1, slice2) == result

    @pytest.mark.parametrize('slice1,slice2,result', [
        [sl[20:30, 40:50], sl[25:27, 42:44], sl[5:7, 2:4]],
        [sl[:30, :50], sl[25:27, 42:44], sl[25:27, 42:44]],
        [sl[:30, :50], sl[:27, :44], sl[:27, :44]],
    ])
    def test_change_origin(self, slice1, slice2, result):
        # Given two slices one of which is subslice of another
        # Compute slice that can be used to retrive result from parent slice (adjust origin point)
        subslice = change_origin(slice1, slice2)
        assert subslice == result
