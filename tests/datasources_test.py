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
    Decorator data source provides cachable requests to underlying src
    """
    def __init__(self, orig):
        self._orig = orig
        self._cache = {}

    def dtype(self):
        return self._orig.dtype()

    def request(self, slicing):
        if slicing not in self._cache:
            self._cache[slicing] = self._orig.request(slicing)
        return self._cache[slicing]


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
        src.request(slicing)
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
        [sl[1:40, 1:40], sl[1:39, 1:40], True],
    ])
    def test_covers(self, slice1, slice2, result):
        def covers(slicing, other):
            assert len(slicing) == len(other)
            for slice_, other_slice in zip(slicing, other):
                if slice_.stop < other_slice.stop or slice_.start > other_slice.start:
                    return False
            return True
        assert covers(slice1, slice2) == result
