import json
import os
from warnings import warn

import numpy as np
try:
    from pandas import Series
except ImportError:
    warn("could not import pandas, conversion disabled")
    Series = NotImplemented

from .utils import getTerminalSize, _MEMMAP_EXT, _ATTR_EXT, _ACCESS_MODES


class MemmapColumn(np.memmap):
    """
    Table column backed by a self-descriptive binary numpy memory map. Inherits
    the properties of a numpy memory map, including slicing and numerical
    methods. Additionally implements user defined descriptive data attributes.

    Parameters:
    -----------
    path : string
        File path (without file extension) at which the underlying memory map
        is stored.
    dtype : string or type
        Defining the column data type, must be supported by numpy (ignored
        unless in w+ mode).
    shape : int or tuple
        Valid integer describing the column length or tuple describing
        the axis dimensions of a multidimensional column (ignored unless in w+
        mode).
    mode : char
        Must be 'r' (read-only), 'r+' (read+write)  or 'w+' (overwrite
        existing).
    """

    _attr = None

    def __new__(cls, path, dtype=None, shape=None, mode="r"):
        # check the access mode
        if mode not in _ACCESS_MODES:
            message = "mode must be one of {:}"
            raise ValueError(message.format(str(_ACCESS_MODES)))

        if mode == "w+":
            # check the input dtype
            if type(dtype) not in (str, type, np.dtype):
                print(path, dtype, shape, mode)
                message = "expected dtype of type {:}, {:} or {:} but got {:}"
                raise TypeError(
                    message.format(
                        str(np.dtype), str, str(type), str(type(dtype))))
            elif type(dtype) is np.dtype:
                dtype = dtype.str
            else:
                dtype = np.dtype(dtype).str
            # check the shape
            message = "shape must be a positive integer or a tuple of integers"
            if type(shape) is int:
                if shape <= 0:
                    raise ValueError(message)
                shape = (shape,)
            elif type(shape) is tuple:
                if not all(type(i) is int for i in shape):
                    raise ValueError(message)
                if not all(i >= 0 for i in shape):
                    raise ValueError(message)
            else:
                raise ValueError(message)
            # create a JSON metadata
            meta_dict = {"dtype": dtype, "shape": shape, "attr": None}
            with open(path + _ATTR_EXT, "w") as f:
                json.dump(meta_dict, f)

        else:
            # load existing data
            if not os.path.exists(path + _ATTR_EXT):
                message = "target '{:}' not found"
                raise OSError(message.format(path))
            else:
                if dtype is not None:
                    print("WARNING: ignoring 'dtype' parameter")
                if shape is not None:
                    print("WARNING: ignoring 'shape' parameter")
                # load the meta data from the companion JSON file
                with open(path + _ATTR_EXT) as f:
                    meta_dict = json.load(f)
                dtype = meta_dict.pop("dtype")
                shape = tuple(meta_dict.pop("shape"))

        try:  # initialize the memory map and the write the meta data
            instance = super().__new__(
                cls, path + _MEMMAP_EXT, np.dtype(dtype), mode, shape=shape)
            # assign the attributes
            instance.__dict__["_attr"] = meta_dict.pop("attr")
        except Exception as e:
            # if anything went wrong, delete newly created output
            if mode == "w+":
                for ext in (_ATTR_EXT, _MEMMAP_EXT):
                    try:
                        os.remove(path + ext)
                    except OSError:
                        pass
            raise e
        return instance

    def _value_formatter(self, max_disp=10, padding="0"):
        """
        Formats the data values into a list of equal length strings. If there
        are more then the maximum number of values, the middle part of data is
        omitted with an ellipsis (...).

        Parameters:
        -----------
        max_disp : int
            Maximum number of items to display without using an ellipsis.
        padding : char
            Character to aling floating point numbers with trailing zeros.

        Returns:
        --------
        representation : list of str
            List of equal width string formatted values
        width : int
            Length of the strings in 'values'.
        """
        use_ellipsis = True
        syb_ellipsis = "..."
        try:
            n_items = len(self)  # TypeError on scalars
            # collect the values to display
            if max_disp >= n_items:
                idx = list(range(0, n_items))  # all elements
                use_ellipsis = False
            else:
                idx = [
                    *range(0, (max_disp + 1) // 2),  # first half of elements
                    *range(n_items - max_disp // 2, n_items)]  # final half
            # get the values, if the columns has multiple dimensions, get only
            # the first element from all these dimensions
            values = [self[i].flat[0] for i in idx]
            # format the values
            if self.dtype.kind == "f":
                # choose the best float formatter
                absolute_max = np.abs(max(values))
                if (1e-4 > absolute_max) or (absolute_max > 1e4):  # scientific
                    representation = []
                    for v in values:
                        val_str = "{:.4e}".format(v)
                        man, exp = val_str.split("e")
                        exp = exp[0] + "0" + exp[1:]
                        representation.append("e".join([man, exp]))
                else:  # use fixed floating point
                    representation = []
                    for v in values:
                        val_str = "{:.6f}".format(v)
                        num, dec = val_str.split(".")
                        # truncate trailing zeros
                        dec = dec[0] + dec[1:].rstrip("0")
                        representation.append([num, dec])
                    # rund second pass to align the floating points
                    max_decimal = max(len(dec) for num, dec in representation)
                    # join to a single floating point number
                    for i, (num, dec) in enumerate(representation):
                        representation[i] = ".".join(
                            [num, dec.ljust(max_decimal, padding)])
            else:
                representation = [str(v) for v in values]
            # if the elements are multidimensional, decorate with with braces
            # to indicate the number of dimensions
            if self.ndim > 1:
                braces_left = "[" * (self.ndim - 1)
                braces_right = "]" * (self.ndim - 1)
                for i, rep in enumerate(representation):
                    representation[i] = "{:}{:}, {:}{:}".format(
                        braces_left, rep, syb_ellipsis, braces_right)
            # get the maximum length of the representations
            length = max(len(rep) for rep in representation)
            length = max(length, len(syb_ellipsis))  # minimal possible width
            # make all strings equal length
            representation = [rep.rjust(length) for rep in representation]
            # insert the ellipsis
            if use_ellipsis:
                representation.insert(
                    (max_disp + 1) // 2, syb_ellipsis.rjust(length))
        except ValueError:
            representation = ["[]"]
            length = 0
        except TypeError:  # formatting scalars (e.g. after slicing  ...)
            representation = [super().__repr__()]
            length = len(representation)
        return representation, length

    def _update_shape(self, new_shape):
        meta_dict = dict(
            dtype=self.dtype.str, shape=new_shape, attr=self.attr)
        with open(self.filename.replace(_MEMMAP_EXT, _ATTR_EXT), "w") as f:
            json.dump(meta_dict, f)

    def __repr__(self):
        try:  # this will succeed if the data is not a scalar
            # header
            string = "<{:} object at {:}\n".format(
                self.__class__.__name__, hex(id(self)))
            # format the values
            representation = self._value_formatter()[0]
            for rep in representation:
                string += " {:}\n".format(rep)  # indent matching the header
            # footer
            string += " shape: {:}, dtype: {:},\n memmap: {:} >".format(
                str(self.shape), str(self.dtype), self.filename)
        except TypeError:  # formatting scalars (e.g. after slicing  ...)
            string = super().__repr__()
        return string

    def __str__(self):
        return super().__str__()

    @property
    def attr(self):
        """
        Obtain user-defined data attributes. These are only preserved, if the
        instance is backed with a valid memory map and an attribute file.

        Returns:
        --------
        attr : None, bool, float, int, str, list, dict
            Data attribute.
        """
        if self.filename is not None:
            with open(self.filename.replace(_MEMMAP_EXT, _ATTR_EXT)) as f:
                meta_dict = json.load(f)
            self._attr = meta_dict.pop("attr")
        return self._attr
    
    @attr.setter
    def attr(self, attribute):
        """
        Assign a data attribute. Attributes are flushed to disk immedately if a
        valid memory map and an attribute file exists.

        Parameters:
        --------
        attr : None, bool, float, int, str, list, dict
            Data attribute.
        """
        self.flush()  # prevent data loss if the serialisation fails
        # allow manipulation only in write mode
        if self.mode == "r":
            raise OSError("write access not permitted")
        # verify that the data can be serialised
        attr_str = json.dumps({"attr": attribute})  # raises an error otherwise
        self._attr = attribute
        # serialise attributes if there is a file backend
        if self.filename is not None:
            metadata = dict(dtype=self.dtype.str, shape=self.shape)
            # serialise before modifying the meta data
            meta_str = json.dumps(metadata)
            # join the JSON strings and write to disk
            json_str = "{{{:}, {:}}}".format(meta_str[1:-1], attr_str[1:-1])
            with open(self.filename.replace(_MEMMAP_EXT, _ATTR_EXT), "w") as f:
                f.write(json_str)

    def to_series(self, index=None, dtype=None, name=None):
        """
        Convert the data to a pandas.Series object.

        Parameters:
        -----------
        index : array-like
            Values must be hashable and have the same length as data. Non-
            unique index values are allowed. Will default to RangeIndex
            (0, 1, 2, …, n) if not provided. If both a dict and index sequence
            are used, the index will override the keys found in the dict.
        dtype : str or numpy.dtype
            Data type for the output Series. If not specified, this will be
            inferred from data. See the user guide for more usages.
        name : str
            The name to give to the Series.
        
        Returns:
        --------
        series : pandas.Series
            Series containing the memory mapped data.
        """
        if Series is NotImplemented:
            raise ImportError("'pandas' installation is not available")
        elif self.ndim > 1:
            message = "conversion to pandas is only possibe for 1-dim data"
            raise ValueError(message)
        series = Series(self, index, dtype, name)
        return series
