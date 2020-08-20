# mmaptable

Self-describing columnar data storage system that is backed by memory mapping.

A `MemmapTable` object is a table that contains a collection of binary columns with
equal length. The table can be resized and columns can be added, copied, renamed or
deleted dynamically. By design, data is preferentially accessed column-wise

## Columns

Each column is represented by a `MemmapColumn`, a memmory mapped binary array based on
`numpy` arrays. Columns therefore behave like ordinary numpy arrays and support all
fixed-size datatypes implemented in numpy. Additionally, `MemmapColumn` implements
attributes that can hold arbitrary Python objects that are JSON serialisable (i.e.
`int`, `float`, `str`, `bool`, `list`, `dict`).

## Table

`MemmapTable` behaves similar to `numpy` arrays with fields and support the same slicing
and assignement logic. The most important difference is, that they are not contiguous in
memmory. Instead, a table is an ordinary filesystem directory in which the memmory-
mapped columns are stored. The column names are determined form the memmory map file
names, similar to `HDF5` paths with sub-directories (=`HDF5`-group) can be used.

For example creating a table at path `/my/table` with columns `index`, `coordinates/x`
and `date` creates the following structure in the file system:
```
/my/table
 ├─ index.npy
 ├─ index.attr
 ├─ coordinates
 │   ├─ x.npy
 │   └─ x.attr
 ├─ date.npy
 └─ data.attr
```

## Example code

```python
import mmaptable
import numpy as np

# create a new table
with mmaptable.MemmapTable("mytable", nrows=100, mode="w+") as tab:

    # add an int32 type column
    int_col = tab.add_column("int_col", dtype="i4")
    int_col.attr = "my first column"
    print(int_col.attr)
    int_col[:] = list(range(len(tab))
    int_col[0] = int_col[-1]

    # add an float64 type column
    float_col = tab.add_column("float_col", dtype="f8", attr="my second column")
    col[:] = 2.0
    print(col.sum())

    # show the newly created table
    print(tab)
    
    # get columns
    print(tab["float_col"])  # the column with name "float_col"

    # get slices
    new_slice = tab[5:10]
    print(new_slice)
    print(tab[["float_col"]])  # table with single column
    print(type(tab), type(new_slice))

    # slices are assignable with numpy arrays of the correct type
    new_data = np.zeros(10, dtype=tab.dtype)
    tab[5:15] = new_data
    # slices do not implement column operations
    try:
        new_slice.add_column("test_column", "bool")
    except Exception as e:
        print(e)
        
    # tables can be resized
    tab.resize(10)  # this will discard the elements 10-99
    tab.resize(20)  # this will add 10 new elements in all columns initialized to 0
    
    # tables can be converted to numpy.array or pandas.DataFrames
    tab.to_records()
    tab.to_dataframe()
```
