import os
import shutil
from collections import OrderedDict, Counter
from warnings import warn
from sys import stdout

import numpy as np
try:
    from pandas import DataFrame
except ImportError:
    warn("could not import pandas, conversion disabled")
    DataFrame = NotImplemented

from .utils import getTerminalSize
from .column import MemmapColumn
from .mathexpression import MathTerm


class MemmapTableSlice:
    """
    Slice of a MemmapTable with limited data access returned by MemmapTable
    when creating slices. While item assignement is allowed if writing on the
    parent table is permitted, columns cannot be added, renamed, copied or
    deleted. If the parent MemmapTable is closed, the data of the slice is no
    longer available.

    Parameters:
    -----------
    parent : MemmapTable
        Reference to the parent MemmapTable from which the data originates.
    columns : OrderedDict
        Internal column buffers of the parent Table that are passed on the
        sliced.
    """

    _parent = None

    def __init__(self, parent, columns):
        self._parent = parent
        self._root = parent.root
        self._mode = parent.mode
        # finally register the subset of columns and update the lenght of
        # the child table
        self._columns = columns
        self._len = self._check_column_length(columns)

    @staticmethod
    def _check_column_length(coldict) -> int:
        """
        Verify that all columns have the same number of rows to detect
        potential corruption of the meta data.

        Parameters:
        -----------
        coldict : dict
            Mapping of column names to column buffers (MemmapColumn).
        
        Returns:
        --------
        length :  int
            Number of rows of the table.
        """
        if len(coldict) == 0:
            length = 0  # table has no columns
        else:
            try:
                lengths = {col: len(data) for col, data in coldict.items()}
            except TypeError:
                length = 1  # columns buffers are scalars
            else:
                # verify that all column buffers have the same length
                length, n_common = Counter(lengths.values()).most_common()[0]
                if n_common != len(lengths):
                    for col, length in lengths.items():
                        message = "'{:}' does not match common length {:d}"
                        raise ValueError(message.format(col, length))
        return length

    def _check_state(self) -> None:
        """
        Check whether the underlying memory map file pointers are closed and
        raise an error if so.
        """
        if self.closed:
            raise ValueError("I/O operation on closed table")

    def _value_formatter(self, max_disp=10) -> tuple:
        """
        Formats the data of the columns into a list of equal length strings. If
        there is insufficient space along either axis, the middle part of
        rows and the right-most columns are omitted with an ellipsis (...).

        Parameters:
        -----------
        max_disp : int
            Maximum number of items to display without using an ellipsis.

        Returns:
        --------
        representation : list of str
            List of equal width string formatted values
        width : int
            Length of the strings in 'values'.
        """
        syb_ellipsis = "..."
        # reserve some space needed in a typical ipython window
        terminal_width, terminal_height = getTerminalSize()
        max_disp = min(terminal_height - 9, max_disp)
        max_disp = max(max_disp, 1)  # minimum to show
        # format the table content
        if len(self._columns) > 0:
            # format the column values their headers by formatting them to a
            # list of right justified strings with equal length
            column_representations = []
            column_widths = []
            for colname, data in self._columns.items():
                try:  # table with multiple rows
                    representation, width = data._value_formatter(max_disp)
                except AttributeError:  # table with scalar column values
                    representation = [str(data)]
                    width = len(representation[0])
                width = max(  # width required for column
                    width, len(colname), len(syb_ellipsis))
                representation.insert(0, colname)  # place header
                column_representations.append([  # justify at common length
                    " {:} ".format(rep.rjust(width))  # add spaces as separator
                    for rep in representation])
                column_widths.append(width + 2)  # includes separator
            n_rows = len(column_representations[0])
            # truncate the table if it is too wide
            # Add columns until the terminal window space runs out. One column
            # is always visible, otherwise buy a bigger screen.
            use_ellipsis = False
            display_columns = [0]
            current_width = column_widths[0]
            for col_idx in range(1, len(column_widths)):
                # stop if the space is insufficient
                if current_width + column_widths[col_idx] > terminal_width:
                    use_ellipsis = True
                    break
                # include the current column
                current_width += column_widths[col_idx]
                display_columns.append(col_idx)
            # replace the last visible column with ellipses
            if use_ellipsis:
                # place the ellipsis in the remaining space on the right side
                right_side_space = \
                    terminal_width - sum(column_widths[:col_idx - 1])
                width = min(right_side_space, 13)
                column_representations[col_idx - 1] = [
                    syb_ellipsis.center(width)] * n_rows
            # join the rows to single strings
            representation = []
            for row_idx in range(n_rows):
                string = ""
                for col_idx in display_columns:
                    string += column_representations[col_idx][row_idx]
                representation.append(string)
            length = len(string)  # all strings have equal length
        else:  # empty table
            representation = []
            length = 0
        return representation, length

    def __repr__(self):
        if self.closed:
            # header
            string = "<{c:} of closed {p:} object at {i:}>".format(
                c=self.__class__.__name__, p=self.parent.__class__.__name__,
                i=hex(id(self)))
        else:
            # header
            string = "<{c:} of {p:} object at {i:}\n".format(
                c=self.__class__.__name__, p=self.parent.__class__.__name__,
                i=hex(id(self)))
            # format the columns
            representation, width = self._value_formatter()
            if width > 0:  # otherwise the table is empty
                for rep in representation:
                    string += rep + "\n"
            # footer
            string += "\n [{r:,d} rows x {d:d} columns]\n".format(
                r=self.shape[1], d=self.shape[0])
            string += " mode: {m:} >".format(m=self.mode)
        return string

    def __len__(self):
        if self.closed:
            return 0
        else:
            return self._len

    def __contains__(self, item):
        return item in self._columns

    def __iter__(self):
        self._check_state()
        for i in range(self._len):
            yield self[i]

    def __getitem__(self, item):
        """
        Select a subset of the table. If the indexing item it is interpreted as
        column name and the corresponding MemmapColumn is returned, otherwise
        a MemmapTableSlice instance is returned.
        If the indexing item is a list of strings, these strings are
        interpreted as column names. The returned object will contain a subset
        of the table columns.
        Any other indexing items will select row-subsets. The returned object
        will contain all of the table columns and the indexing item is used
        to select a subset of elements in each column.
        Note: Returned MemmapTableSlice instances have limited write access.
        """
        self._check_state()
        # selecting a single column by its name -> MemmapColumn
        if type(item) is str:
            subtable = self._columns[item]
        # selecting a set of columns or rows -> MemmapTable
        else:
            # Create a child instance representing the selected subset of data,
            # inheriting its data buffers from this parent instance.
            try:
                # the item must be a list of strings which are the column names
                assert(type(item) is list)
                assert(all(type(entry) is str for entry in item))
                columns = OrderedDict(
                    (col, self._columns[col]) for col in item)
            # selecting row subsets, delegate to numpy slicing
            except AssertionError:
                columns = OrderedDict(
                    (col, data[item]) for col, data in self._columns.items())
            if self._parent is None:
                parent = self
            else:
                parent = self._parent
            subtable = MemmapTableSlice(parent, columns)
        return subtable

    def __setitem__(self, item, value):
        """
        Set values of a subset of items of the table. If selecting items of
        multiple columns, the values must be a numpy.ndarray with matching
        field names and compatible data types.
        """
        self._check_state()
        # make sure the table is not in read-only mode
        if self.mode == "r":
            raise OSError("assignments not permitted")
        if type(value) is type(self):
            for key, data in value._columns.items():
                self._columns[key][item] = data
        elif isinstance(value, np.ndarray):
            for key, data in self._columns.items():
                data[item] = value[key]
        else:
            message = "input type {:} is not supported"
            raise TypeError(message.format(str(type(value))))

    def _ipython_key_completions_(self):
        """
        Used by ipython to infer column names for auto-completion
        """
        return self._columns.keys()

    @property
    def closed(self) -> bool:
        """
        Whether the underlying memory map file pointers are closed.
        """
        try:
            return self._closed
        except AttributeError:
            return self.parent._closed

    @property
    def colnames(self) -> tuple:
        """
        List of the column names.
        """
        return tuple(self._columns.keys())

    @property
    def dtype(self) -> np.dtype:
        """
        Numpy data type of the data table columns.
        """
        type_map = []
        for col, data in self._columns.items():
            if data.ndim == 1:
                type_map.append((col, data.dtype))
            else:
                type_map.append((col, data.dtype, data.shape[1:]))
        return np.dtype(type_map)

    @property
    def itemsize(self) -> int:
        """
        Total number of bytes a row of the table occupies.
        """
        return sum(data.itemsize for data in self._columns.values())

    @property
    def mode(self) -> str:
        """
        Access mode identifier of the table data: r (read-only), r+
        (read-write), a (item assignment), w+ (overwrite)
        """
        return self._mode

    @property
    def nbytes(self) -> int:
        """
        Total number of the bytes the table occupies.
        """
        return sum(data.nbytes for data in self._columns.values())

    @property
    def ndim(self) -> OrderedDict:
        """
        The number of dimensions the data in each column has.
        """
        dim_map = [(col, data.ndim) for col, data in self._columns.items()]
        return OrderedDict(dim_map)

    @property
    def parent(self) -> str:
        """
        Reference to the parent table instance, otherwise None.
        """
        return self._parent

    @property
    def root(self) -> str:
        """
        The path to the root node of the data table.
        """
        return self._root

    @property
    def size(self) -> int:
        """
        Total number of elements in the table, including higher column
        dimensions.
        """
        return sum(data.size for data in self._columns.values())

    @property
    def shape(self) -> tuple:
        """
        The number of columns and rows of the table. This does not reflect the
        true shape of the table since each column can be multidimensional.
        """
        return (len(self._columns), len(self),)

    def row_iter(self, chunksize=10000) -> iter:
        """
        Yields an iterator over chunks of the table rows, optionally showing
        the progress.

        Parameters:
        -----------
        chunksize : int
            Number of rows to select per iteration.

        Yields:
        -------
        start : int
            First index of the chunk.
        end : int
            Final index of the chunk (exclusive).
        """
        self._check_state()
        n_max = len(self)
        # yield first and last index
        # iterate in chunks
        if type(chunksize) is int and chunksize > 0:
            for start in range(0, n_max, chunksize):
                # truncate last chunk to table length
                end = min(n_max, start + chunksize)
                yield start, end
        else:
            message = "chunksize must be None or a positive integer"
            raise TypeError(message)

    def to_dataframe(self, index=None) -> DataFrame:
        """
        Convert the table data to a pands.DataFrame object with the same data
        type and column names.

        Returns:
        --------
        df : pandas.DataFrame
            Table converted to a pandas data frame.
        """
        self._check_state()
        if DataFrame is NotImplemented:
            raise ImportError("'pandas' installation is not available")
        elif any(ndim > 1 for ndim in self.ndim.values()):
            message = "conversion to pandas is only possibe for 1-dim columns"
            raise ValueError(message)
        col_dict = OrderedDict()
        idx_data = None
        for colname, data in self._columns.items():
            if len(data.shape) == 0:  # scalar data (e.g. row slice)
                if colname == index:
                    idx_data = [data]
                else:
                    col_dict[colname] = [data]
            else:
                if colname == index:
                    idx_data = data
                else:
                    col_dict[colname] = data
        df = DataFrame(col_dict, index=idx_data)
        return df

    def to_records(self) -> np.recarray:
        """
        Convert the table data to a numpy.recarray object with the same data
        type and column names converted to field names.

        Returns:
        --------
        recarray : numpy.recarray
            Record array with the same layout as the table data.
        """
        self._check_state()
        recarray = np.recarray((self._len,), dtype=self.dtype)
        for col  in self.colnames:
            recarray[col] = self[col]
        return recarray

    def flush(self) -> None:
        """
        Flush all internal column buffers.
        """
        if self.mode != "r" and not self.closed:
            for col in self._columns.values():
                col.flush()


class MemmapTable(MemmapTableSlice):
    """
    Table backed by self-descriptive binary numpy memory maps. Mimics the
    indexing behaviour of structured numpy arrays and inherits some descriptive
    attributes Data is stored in binary files below a root directory with
    additional descriptive meta data (data type, shape, attributes), see
    MemmapColumn.

    Parameters:
    -----------
    path : string
        File path of the root node.
    nrows : int
        Number of rows at which all (new) columns will be initialized (ignored
        if the table is not empty).
    mode : char
        Must be 'r' (read-only), 'r+' (read+write)  or 'w+' (overwrite
        existing). Note: a (assignment mode) does not allow creating or
        deleting columns, but allows modifying column elements. This is the
        standard mode of all table slices.
    """

    _closed = False
    _parent = None

    def __init__(self, path, nrows=None, mode="r"):
        self._root = os.path.expanduser(os.path.abspath(path))
        # check the requested number of rows
        if nrows is not None:
            message = "'nrows' must be a positive integer"
            if type(nrows) is not int:
                raise ValueError(message)
            elif nrows <= 0:
                raise ValueError(message)
        # check the access mode
        if mode not in MemmapColumn._ACCESS_MODES:
            message = "mode must be one of {:}"
            raise ValueError(message.format(str(MemmapColumn._ACCESS_MODES)))
        self._mode = mode
        if self.mode == "w+":  # wipe existing data
            if os.path.exists(self.root):
                shutil.rmtree(self.root)
            os.makedirs(path)
            # empty table
            self._columns = OrderedDict()
        else:  # load existing data
            if os.path.exists(self.root):
                # load the data columns as memory maps
                self._columns = self._init_data(self.root, self.mode)
            else:
                message = "root path '{:}' not found"
                raise OSError(message.format(path))
        # check or intialize the table length
        if len(self._columns) > 0:
            if nrows is not None:
                print("WARNING: ignoring 'nrows' parameter")
            self._len = self._check_column_length(self._columns)
        else:
            if nrows is None:
                message = "an empty table must be initialized with 'nrows'"
                raise ValueError(message)
            self._len = nrows

    @staticmethod
    def _init_data(rootpath, mode) -> OrderedDict:
        """
        Scan the root node for valid column buffers. Open all buffers with the
        requrested access permissions.

        Parameters:
        -----------
        rootpath : str
            Path of the root node from which the underlying filesystem is
            analysed.
        mode : str
            Access mode identifiere (must be either of r/r+/a/w+).
        
        Returns:
        --------
        columns : OrderedDict
            Mapping of column names to column buffer instances (MemmapColumn).
        """
        columns = OrderedDict()
        for root, _, files in os.walk(rootpath):
            for f in files:
                abspath = os.path.join(root, f)
                path, ext = os.path.splitext(abspath)
                # find numpy binary files with and attached attribute JSON file
                has_memmap_ext = ext == MemmapColumn._MEMMAP_EXT
                attr_exist = os.path.exists(path + MemmapColumn._ATTR_EXT)
                if has_memmap_ext and attr_exist:
                    relpath = os.path.relpath(path, rootpath)
                    # load the memory map
                    columns[relpath] = MemmapColumn(path, mode=mode)
        return columns

    def __repr__(self):
        if self.closed:
            # header
            string = "<closed {c:} object at {i:}>".format(
                c=self.__class__.__name__, i=hex(id(self)))
        else:
            # header
            string = "<{c:} object at {i:}\n".format(
                c=self.__class__.__name__, i=hex(id(self)))
            # format the columns
            representation, width = self._value_formatter()
            if width > 0:  # otherwise the table is empty
                for rep in representation:
                    string += rep + "\n"
            # footer
            string += "\n [{r:,d} rows x {d:d} columns]\n".format(
                r=self.shape[1], d=self.shape[0])
            string += " mode: {m:}\n".format(m=self.mode)
            string += " root: {r:} >".format(r=self.root)
        return string

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def resize(self, new_length) -> None:
        """
        Resize the memory mapped data columns along their first dimension, the
        length of the table. The file pointers must be closed temporarily and
        are reopened. On expansion, new rows will be initialized with zeros,
        on truncation, trailing rows are lost.

        Parameters:
        -----------
        new_length : int
            New number of rows the table with hold.
        """
        self._check_state()
        message = "'new_length' must be a positive integer"
        if type(new_length) is not int:
            raise ValueError(message)
        elif new_length <= 0:
            raise ValueError(message)
        # access mode restrictions
        if self.mode == "r":
            raise OSError("write access not permitted")
        self.flush() # first flush everything to disk
        if self.mode == "w+":
            self._mode = "r+"
        # tear down the table and resize the buffers
        for colname in self.colnames:
            data = self._columns.pop(colname)
            # resize all underlying base memorymaps
            new_size = new_length * int(np.prod(data.shape[1:]))
            new_bytes = data.itemsize * new_size
            data.base.resize(new_bytes)
            data.flush()  # otherwise the memorymaps do not shrink
            # update the memorymap meta data
            new_shape = [new_length]
            new_shape.extend(data.shape[1:])
            data._update_shape(tuple(new_shape))
        # reinitialize
        self._columns = self._init_data(self.root, self.mode)
        if len(self._columns) == 0:  # resizing an empty table
            self._len = new_length
        else:
            self._len = self._check_column_length(self._columns)

    def add_column(
            self, path, dtype, item_shape=None, attr=None,
            overwrite=False) -> MemmapColumn:
        """
        Create a new column at a given path and with given data type. Optional
        arguments specify the shape of higher data dimensions and set data
        attributes.

        Parameters:
        -----------
        path : str
            Column name (= path at which the column buffer is registered).
        dtype : str or type
            String or type compatible with native numpy data types.
        item_shape : tuple
            Shape tuple describing the shape of each column element. A
            tuple (k, l) results in a 3-dim data column with shape (N, k, l),
            where N is the length of the table.
        attr : dict
            Data attributes assigned to the new column buffer. Must be
            serialisable with JSON.
        overwrite : bool
            Whether to overwrite an existing column.

        Returns:
        --------
        column : MemmapColumn
            The newly created column buffer.
        """
        self._check_state()
        # access mode restrictions
        if self.mode == "r":
            raise OSError("write access not permitted")
        # check that the path is contained in the root node
        if ".." in path:
            raise ValueError("relative paths are not permitted")
        # check if the path already exists
        if path in self._columns and not overwrite:
            message = "path '{:}' already exists and overwrite=False"
            raise KeyError(message.format(path))
        # create missing directories in the path
        filepath = os.path.join(self.root, path)
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # generate the shape tuple with extra dimensions of the elements
        shape = [len(self)]
        if item_shape is not None:
            shape.extend(item_shape)
        # create the column and register it internally
        column = MemmapColumn(filepath, dtype, tuple(shape), mode="w+")
        if attr is not None:  # set the optional attributes
            column.attr = attr
        self._columns[path] = column
        return column

    def delete_column(self, path) -> None:
        """
        Delete a single column by removing the internal buffer and deleting the
        memmory mapped data and it's attributes from disk. If the data is
        stored in a sub-path below the table root, recursively delete any empty
        directories along this sub-path.

        path : str
            Name of the column to delete.
        """
        self._check_state()
        if self.mode in ("r", "a+"):
            raise OSError("write access not permitted")
        if type(path) is not str:
            raise TypeError("can only delete single columns")
        # deregister column internally
        memmap = self._columns.pop(path)
        del memmap
        # delete the files from disk
        base_path = os.path.join(self._root, path)
        os.remove(base_path + MemmapColumn._MEMMAP_EXT)
        os.remove(base_path + MemmapColumn._ATTR_EXT)
        # remove empty parent directories
        path_segments = path.split(os.sep)[:-1]  # drop the actual column name
        for i in range(len(path_segments)):
            parent_path = os.path.join(self._root, *path_segments[:i+1])
            # check if this directory is empty
            if len(os.listdir(parent_path)) == 0:
                shutil.rmtree(parent_path)

    def copy_column(self, src, dst) -> None:
        """
        Rename a data column named 'src' to a new name 'dst'. This corresponds
        to moving the underlying memorymap within the file system.

        Parameters:
        -----------
        src : str
            Current name of the column to rename.
        dst : str
            Name to which the column 'src' will be renamed.
        """
        self._check_state()
        self.flush()  # ensure the data on disk is synchronized
        # get the required source attributes
        source_column = self[src]
        dtype = source_column.dtype
        if len(source_column.shape) == 1:
            item_shape = None
        else:
            item_shape = tuple(s for s in source_column.shape[1:])
        attr = source_column.attr
        # create the destination
        dst_data = self.add_column(dst, dtype, item_shape, attr)
        # copy the data by assignment
        for start, end in self.row_iter(int(1e5), verbose=False):
            dst_data[start:end] = source_column[start:end]

    def rename_column(self, src, dst) -> None:
        """
        Copy a data column named 'src' to a new column 'dst'. This corresponds
        to copying the underlying data on the file system.

        Parameters:
        -----------
        src : str
            Current name of the column to rename.
        dst : str
            Name to which the column 'src' will be renamed.
        """
        self._check_state()
        # copy the column to the destination, then delete the source data
        self.copy_column(src, dst)
        self.delete_column(src)

    def close(self) -> None:
        """
        Flush all and close internal column buffers.
        """
        self.flush()
        # close all column buffers
        while len(self._columns) > 0:
            _, data = self._columns.popitem()
            del data
        self._closed = True
