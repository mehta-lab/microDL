from copy import copy
import os
from numcodecs import Blosc
import numpy as np
import zarr

ARRAY_NAME = "arr_0"


class ReaderBase:
    """
    I/O classes for zarr data are directly copied from:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io

    This will be updated if the io parts of waveorder is moved to a stand alone
    python package.
    """

    def __init__(self):
        self.frames = None
        self.channels = None
        self.slices = None
        self.height = None
        self.width = None
        self.dtype = None
        self.mm_meta = None
        self.stage_positions = None
        self.z_step_size = None
        self.channel_names = None

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

    def get_zarr(self, position: int) -> zarr.array:
        pass

    def get_array(self, position: int) -> np.ndarray:
        pass

    def get_image(self, p, t, c, z) -> np.ndarray:
        pass

    def get_num_positions(self) -> int:
        pass


class WriterBase:
    """
    I/O classes for zarr data are directly copied from:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io

    This will be updated if the io of waveorder is moved to a stand alone
    python package.
    ABC for all writer types
    """

    def __init__(self, store, root_path):

        # init common attributes
        self.store = store
        self.root_path = root_path
        self.current_pos_group = None
        self.current_position = None
        self.current_well_group = None
        self.verbose = False
        self.dtype = None

        # set hardcoded compressor
        self.__compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)

        # maps to keep track of hierarchies
        self.rows = dict()
        self.columns = dict()
        self.positions = dict()

    # Silence print statements
    def set_verbosity(self, verbose: bool):
        self.verbose = verbose

    # Initialize zero array
    def init_array(
        self, data_shape, chunk_size, dtype, chan_names, clims, overwrite=False
    ):
        """
        Initializes the zarr array under the current position subgroup.
        array level is called 'arr_0' in the hierarchy.  Sets omero/multiscales metadata based upon
        chan_names and clims
        Parameters
        ----------
        data_shape:         (tuple)  Desired Shape of your data (T, C, Z, Y, X).  Must match data
        chunk_size:         (tuple) Desired Chunk Size (T, C, Z, Y, X).  Chunking each image would be (1, 1, 1, Y, X)
        dtype:              (str or np.dtype) Data Type, i.e. 'uint16' or np.uint16
        chan_names:         (list) List of strings corresponding to your channel names.  Used for OME-zarr metadata
        clims:              (list) list of tuples corresponding to contrast limtis for channel.  OME-Zarr metadata
                                    tuple can be of (start, end, min, max) or (start, end)
        overwrite:          (bool) Whether or not to overwrite the existing data that may be present.
        Returns
        -------
        """
        self.dtype = np.dtype(dtype)

        self.set_channel_attributes(chan_names, clims)
        self.current_pos_group.zeros(
            ARRAY_NAME,
            shape=data_shape,
            chunks=chunk_size,
            dtype=dtype,
            compressor=self.__compressor,
            overwrite=overwrite,
        )

    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array
        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index slice of the time dimension
        :param c: (list), index or index slice of the channel dimension
        :param z: (list), index or index slice of the z dimension
        """
        shape = np.shape(data)

        if self.current_pos_group.__len__() == 0:
            raise ValueError("Array not initialized")

        if not isinstance(t, int) and not isinstance(t, slice):
            raise TypeError("t specification must be either int or slice")

        if not isinstance(c, int) and not isinstance(c, slice):
            raise TypeError("c specification must be either int or slice")

        if not isinstance(z, int) and not isinstance(z, slice):
            raise TypeError("z specification must be either int or slice")

        if isinstance(t, int) and isinstance(c, int) and isinstance(z, int):

            if len(shape) > 2:
                raise ValueError("Index dimensions exceed data dimensions")
            else:
                self.current_pos_group[ARRAY_NAME][t, c, z] = data

        else:
            self.current_pos_group[ARRAY_NAME][t, c, z] = data

    def create_channel_dict(self, chan_name, clim=None, first_chan=False):
        """
        This will create a dictionary used for OME-zarr metadata.  Allows custom contrast limits and channel
        names for display.  Defaults everything to grayscale.
        Parameters
        ----------
        chan_name:          (str) Desired name of the channel for display
        clim:               (tuple) contrast limits (start, end, min, max)
        first_chan:         (bool) whether or not this is the first channel of the dataset (display will be set to active)
        Returns
        -------
        dict_:              (dict) dictionary adherent to ome-zarr standards
        """
        if chan_name == "Retardance":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 1000.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 100.0
        elif chan_name == "Orientation":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else np.pi
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else np.pi

        elif chan_name == "Phase3D":
            min = clim[2] if clim else -10.0
            max = clim[3] if clim else 10.0
            start = clim[0] if clim else -0.2
            end = clim[1] if clim else 0.2

        elif chan_name == "BF":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 5.0

        elif chan_name == "S0":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 1.0

        elif chan_name == "S1":
            min = clim[2] if clim else 10.0
            max = clim[3] if clim else -10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == "S2":
            min = clim[2] if clim else -10.0
            max = clim[3] if clim else 10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == "S3":
            min = clim[2] if clim else -10
            max = clim[3] if clim else 10
            start = clim[0] if clim else -1.0
            end = clim[1] if clim else 1.0

        else:
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 65535.0

        dict_ = {
            "active": first_chan,
            "coefficient": 1.0,
            "color": "FFFFFF",
            "family": "linear",
            "inverted": False,
            "label": chan_name,
            "window": {"end": end, "max": max, "min": min, "start": start},
        }

        return dict_

    def create_row(self, idx, name=None):
        """
        Creates a row in the hierarchy (first level below zarr store). Option to name
        this row.  Default is Row_{idx}.  Keeps track of the row name + row index for later
        metadata creation
        Parameters
        ----------
        idx:            (int) Index of the row (order in which it is placed)
        name:           (str) Optional name to replace default row name
        Returns
        -------
        """
        row_name = f"Row_{idx}" if not name else name
        row_path = os.path.join(self.root_path, row_name)

        # check if the user is trying to create a row that already exsits
        if os.path.exists(row_path):
            raise FileExistsError(
                f"A row subgroup with the name {row_name} already exists"
            )
        else:
            self.store.create_group(row_name)
            self.rows[idx] = row_name

    def create_column(self, row_idx, idx, name=None):
        """
        Creates a column in the hierarchy (second level below zarr store, one below row). Option to name
        this column.  Default is Col_{idx}.  Keeps track of the column name + column index for later
        metadata creation
        Parameters
        ----------
        row_idx:        (int) Index of the row to place the column underneath
        idx:            (int) Index of the column (order in which it is placed)
        name:           (str) Optional name to replace default column name
        Returns
        -------
        """
        col_name = f"Col_{idx}" if not name else name
        row_name = self.rows[row_idx]
        col_path = os.path.join(os.path.join(self.root_path, row_name), col_name)

        # check to see if the user is trying to create a row that already exists
        if os.path.exists(col_path):
            raise FileExistsError(
                f"A column subgroup with the name {col_name} already exists"
            )
        else:
            self.store[self.rows[row_idx]].create_group(col_name)
            self.columns[idx] = col_name

    def open_position(self, position: int):
        """
        Opens a position based upon the position index.  It will navigate the rows/column to
        find where this position is based off of the generation position map which keeps track
        of this information.  It will set current_pos_group to this position for writing the data
        Parameters
        ----------
        position:           (int) Index of the position you wish to open
        Returns
        -------
        """
        # get row, column, and path to the well
        row_name = self.positions[position]["row"]
        col_name = self.positions[position]["col"]
        well_path = os.path.join(os.path.join(self.root_path, row_name), col_name)

        # check to see if this well exists (row/column)
        if os.path.exists(well_path):
            pos_name = self.positions[position]["name"]
            pos_path = os.path.join(well_path, pos_name)

            # check to see if the position exists
            if os.path.exists(pos_path):

                if self.verbose:
                    print(f"Opening subgroup {row_name}/{col_name}/{pos_name}")

                # update trackers to note the current status of the writer
                self.current_pos_group = self.store[row_name][col_name][pos_name]
                self.current_well_group = self.store[row_name][col_name]
                self.current_position = position

            else:
                raise FileNotFoundError(
                    f"Could not find zarr position subgroup at {row_name}/{col_name}/{pos_name}\
                                                    Check spelling or create position subgroup with create_position"
                )
        else:
            raise FileNotFoundError(
                f"Could not find zarr position subgroup at {row_name}/{col_name}/\
                                                Check spelling or create column/position subgroup with create_position"
            )

    def set_root(self, root):
        """
        set the root path of the zarr store.  Used in the main writer class.
        Parameters
        ----------
        root:               (str) path to the zarr store (folder ending in .zarr)
        Returns
        -------
        """
        self.root_path = root

    def set_store(self, store):
        """
        Sets the zarr store.  Used in the main writer class
        Parameters
        ----------
        store:              (Zarr StoreObject) Opened zarr store at the highest level
        Returns
        -------
        """
        self.store = store

    def get_zarr(self):
        return self.current_pos_group

    def set_channel_attributes(self, chan_names, clims=None):
        """
        A method for creating ome-zarr metadata dictionary.
        Channel names are defined by the user, everything else
        is pre-defined.
        Parameters
        ----------
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.
        clims:          (list of tuples) contrast limits to display for every channel
        """

        rdefs = {"defaultT": 0, "model": "color", "projection": "normal", "defaultZ": 0}

        multiscale_dict = [{"datasets": [{"path": ARRAY_NAME}], "version": "0.1"}]
        dict_list = []

        if clims and len(chan_names) < len(clims):
            raise ValueError(
                "Contrast Limits specified exceed the number of channels given"
            )

        for i in range(len(chan_names)):
            if clims:
                if len(clims[i]) == 2:
                    if "float" in self.dtype.name:
                        clim = (float(clims[i][0]), float(clims[i][1]), -1000, 1000)
                    else:
                        info = np.iinfo(self.dtype)
                        clim = (
                            float(clims[i][0]),
                            float(clims[i][1]),
                            info.min,
                            info.max,
                        )
                elif len(clims[i]) == 4:
                    clim = (
                        float(clims[i][0]),
                        float(clims[i][1]),
                        float(clims[i][2]),
                        float(clims[i][3]),
                    )
                else:
                    raise ValueError("clim specification must a tuple of length 2 or 4")

            first_chan = True if i == 0 else False
            if not clims or i >= len(clims):
                dict_list.append(
                    self.create_channel_dict(chan_names[i], first_chan=first_chan)
                )
            else:
                dict_list.append(
                    self.create_channel_dict(chan_names[i], clim, first_chan=first_chan)
                )

        full_dict = {
            "multiscales": multiscale_dict,
            "omero": {"channels": dict_list, "rdefs": rdefs, "version": 0.1},
        }

        self.current_pos_group.attrs.put(full_dict)

    def init_hierarchy(self):
        pass

    def create_position(self, position: int, name: str):
        pass


class ZarrReader(ReaderBase):
    """
    I/O classes for zarr data are directly copied from:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io

    Reader for HCS ome-zarr arrays.  OME-zarr structure can be found here: https://ngff.openmicroscopy.org/0.1/
    Also collects the HCS metadata so it can be later copied.
    """

    def __init__(self, zarrfile: str):
        super().__init__()

        # zarr files (.zarr) are directories
        if not os.path.isdir(zarrfile):
            raise ValueError("file does not exist")
        try:
            self.store = zarr.open(zarrfile, "r")
        except:
            raise FileNotFoundError(
                "Path: {} is not a valid zarr store".format(zarrfile)
            )

        try:
            row = self.store[list(self.store.group_keys())[0]]
            col = row[list(row.group_keys())[0]]
            pos = col[list(col.group_keys())[0]]
            self.arr_name = list(pos.array_keys())[0]
        except IndexError:
            raise IndexError("Incompatible zarr format")

        self.plate_meta = self.store.attrs.get("plate")
        self._get_rows()
        self._get_columns()
        self._get_wells()
        self.position_map = self._get_positions()

        # structure of zarr array
        (self.frames, self.channels, self.slices, self.height, self.width) = self.store[
            self.position_map[0]["well"]
        ][self.position_map[0]["name"]][self.arr_name].shape
        self.positions = len(self.position_map)
        self.channel_names = []
        self.stage_positions = 0
        self.z_step_size = None

        # initialize metadata
        self.mm_meta = None

        try:
            self._set_mm_meta()
        except TypeError:
            self.mm_meta = None

        self._generate_hcs_meta()

        # get channel names from omero metadata if no MM meta present
        if len(self.channel_names) == 0:
            self._get_channel_names()

    def _get_rows(self):
        """
        Function to get the rows of the zarr hierarchy from HCS metadata
        """
        rows = []
        for row in self.plate_meta["rows"]:
            rows.append(row["name"])
        self.rows = rows

    def _get_columns(self):
        """
        Function to get the columns of the zarr hierarchy from HCS metadata
        """
        columns = []
        for column in self.plate_meta["columns"]:
            columns.append(column["name"])
        self.columns = columns

    def _get_wells(self):
        """
        Function to get the wells (Row/Col) of the zarr hierarchy from HCS metadata
        """
        wells = []
        for well in self.plate_meta["wells"]:
            wells.append(well["path"])
        self.wells = wells

    def _get_positions(self):
        """
        Gets the position names and paths from HCS metadata
        """
        position_map = dict()
        idx = 0
        # Assumes that the positions are indexed in the order of Row-->Well-->FOV
        for well in self.wells:
            for pos in self.store[well].attrs.get("well").get("images"):
                name = pos["path"]
                position_map[idx] = {"name": name, "well": well}
                idx += 1
        return position_map

    def _generate_hcs_meta(self):
        """
        Pulls the HCS metadata and organizes it into a dictionary structure
        that can be easily read by the WaveorderWriter.
        """
        self.hcs_meta = dict()
        self.hcs_meta["plate"] = self.plate_meta

        well_metas = []
        for well in self.wells:
            meta = self.store[well].attrs.get("well")
            well_metas.append(meta)

        self.hcs_meta["well"] = well_metas

    def _set_mm_meta(self):
        """
        Sets the micromanager summary metadata based on MM version
        """
        self.mm_meta = self.store.attrs.get("Summary")
        mm_version = self.mm_meta["MicroManagerVersion"]

        if mm_version != "pycromanager":
            if "beta" in mm_version:
                if self.mm_meta["Positions"] > 1:
                    self.stage_positions = []

                    for p in range(len(self.mm_meta["StagePositions"])):
                        pos = self._simplify_stage_position_beta(
                            self.mm_meta["StagePositions"][p]
                        )
                        self.stage_positions.append(pos)

            else:
                if self.mm_meta["Positions"] > 1:
                    self.stage_positions = []

                    for p in range(self.mm_meta["Positions"]):
                        pos = self._simplify_stage_position(
                            self.mm_meta["StagePositions"][p]
                        )
                        self.stage_positions.append(pos)

        self.z_step_size = self.mm_meta["z-step_um"]

    def _get_channel_names(self):

        well = self.hcs_meta["plate"]["wells"][0]["path"]
        pos = self.hcs_meta["well"][0]["images"][0]["path"]

        omero_meta = self.store[well][pos].attrs.asdict()["omero"]

        for chan in omero_meta["channels"]:
            self.channel_names.append(chan["label"])

    def _simplify_stage_position(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys

        :param dict stage_pos: Dictionary containing a single position's device info
        :return dict out: Flattened dictionary
        """
        out = copy(stage_pos)
        out.pop("DevicePositions")
        for dev_pos in stage_pos["DevicePositions"]:
            out.update({dev_pos["Device"]: dev_pos["Position_um"]})
        return out

    def _simplify_stage_position_beta(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys
        for MM2.0 Beta versions

        :param dict stage_pos: Dictionary containing a single position's device info
        :return dict new_dict: Flattened dictionary
        """
        new_dict = {}
        new_dict["Label"] = stage_pos["label"]
        new_dict["GridRow"] = stage_pos["gridRow"]
        new_dict["GridCol"] = stage_pos["gridCol"]

        for sub in stage_pos["subpositions"]:
            values = []
            for field in ["x", "y", "z"]:
                if sub[field] != 0:
                    values.append(sub[field])
            if len(values) == 1:
                new_dict[sub["stageName"]] = values[0]
            else:
                new_dict[sub["stageName"]] = values

        return new_dict

    def get_image_plane_metadata(self, p, c, z):
        """
        For the sake of not keeping an enormous amount of metadata, only the microscope conditions
        for the first timepoint are kept in the zarr metadata during write.  User can only query image
         plane metadata at p, c, z

        :param int p: Position index
        :param int c: Channel index
        :param int z: Z-slice index
        :return dict metadata: Image Plane Metadata at given coordinate w/ T = 0
        """
        coord_str = f"({p}, 0, {c}, {z})"
        return self.store.attrs.get("ImagePlaneMetadata").get(coord_str)

    def get_zarr(self, position):
        """
        Returns the position-level zarr group array (not in memory)

        :param int position: Position index
        :return ZarrArray Zarr array containing the (T, C, Z, Y, X) array at given position
        """
        pos_info = self.position_map[position]
        well = pos_info["well"]
        pos = pos_info["name"]
        return self.store[well][pos][self.arr_name]

    def get_array(self, position):
        """
        Gets the (T, C, Z, Y, X) array at given position

        :param int position: Position index
        :return np.array pos: Array of size (T, C, Z, Y, X) at specified position
        """
        pos = self.get_zarr(position)
        return pos[:]

    def get_image(self, p, t, c, z):
        """
        Returns the image at dimension P, T, C, Z

        :param int p: Index of the position dimension
        :param int t: Index of the time dimension
        :param int c: Index of the channel dimension
        :param int z: Index of the z dimension
        :return np.array image: Image at the given dimension of shape (Y, X)
        """
        pos_idx = p
        if self.positions == 1:
            pos_idx = 0
        pos = self.get_zarr(pos_idx)
        return pos[t, c, z]

    def get_num_positions(self) -> int:
        return self.positions


class ZarrWriter:
    """
    I/O classes for zarr data are directly copied from:
    https://github.com/mehta-lab/waveorder/tree/master/waveorder/io

    given stokes or physical data, construct a standard hierarchy in zarr for output
        should conform to the ome-zarr standard as much as possible
    """

    __builder = None
    __save_dir = None
    __root_store_path = None
    __builder_name = None
    __current_zarr_group = None
    store = None

    current_group_name = None
    current_position = None

    def __init__(
        self, save_dir: str = None, hcs_meta: dict = None, verbose: bool = False
    ):

        self.verbose = verbose
        self.hcs_meta = hcs_meta

        if os.path.exists(save_dir) and save_dir.endswith(".zarr"):
            print(f"Opening existing store at {save_dir}")
            self._open_zarr_root(save_dir)
        else:
            self._check_is_dir(save_dir)

        # initialize Default writer
        self.sub_writer = DefaultZarr(self.store, self.__root_store_path)

        if self.verbose:
            self.sub_writer.set_verbosity(self.verbose)

    def _check_is_dir(self, path):
        """
        directory verification
        assigns self.__save_dir

        :param str path: Directory path
        """
        if os.path.isdir(path) and os.path.exists(path):
            self.__save_dir = path
        else:
            print(f"No existing directory found. Creating new directory at {path}")
            os.mkdir(path)
            self.__save_dir = path

    def _open_zarr_root(self, path):
        # TODO: Use case where user opens an already HCS-store?
        """
        Change current zarr to an existing store
        if zarr doesn't exist, raise error

        :param str path: Path to store. Must end in .zarr
        """
        if os.path.exists(path):
            assert path.endswith(
                ".zarr"
            ), "Path must en in .zarr. Current path: {}".format(path)
            try:
                self.store = zarr.open(path)
                self.__root_store_path = path
            except:
                raise FileNotFoundError(
                    "Path: {} is not a valid zarr store".format(path)
                )
        else:
            raise FileNotFoundError(
                f"No store found at {path}, check spelling or create new store with create_zarr"
            )

    def create_zarr_root(self, name):
        """
        Method for creating the root zarr store.
        If the store already exists, it will raise an error.
        Name corresponds to the root directory name (highest level) zarr store.

        :param str name: Name of the zarr store.
        """
        if not name.endswith(".zarr"):
            name = name + ".zarr"

        zarr_path = os.path.join(self.__save_dir, name)
        if os.path.exists(zarr_path):
            raise FileExistsError("A zarr store with this name already exists")

        # Creating new zarr store
        self.store = zarr.open(zarr_path)
        self.__root_store_path = zarr_path
        self.sub_writer.set_store(self.store)
        self.sub_writer.set_root(self.__root_store_path)
        self.sub_writer.init_hierarchy()

    def init_array(
        self,
        position,
        data_shape,
        chunk_size,
        chan_names,
        dtype="float32",
        clims=None,
        position_name=None,
        overwrite=False,
    ):
        """
        Creates a subgroup structure based on position index.  Then initializes the zarr array under the
        current position subgroup.  Array level is called 'array' in the hierarchy.

        :param int position: Position index upon which to initialize array
        :param tuple data_shape: Desired Shape of your data (T, C, Z, Y, X).  Must match data
        :param tuple chunk_size: Desired Chunk Size (T, C, Z, Y, X).  Chunking each image would be (1, 1, 1, Y, X)
        :param str dtype: Data Type, i.e. 'uint16'
        :parm list chan_names: List of strings corresponding to your channel names.  Used for OME-zarr metadata
        :param list clims: List of tuples corresponding to contrast limtis for channel.  OME-Zarr metadata
        :param bool overwrite: Whether or not to overwrite the existing data that may be present.
        """
        pos_name = position_name if position_name else f"Pos_{position:03d}"

        # Make sure data matches OME zarr structure
        if len(data_shape) != 5:
            raise ValueError(
                "Data shape must be (T, C, Z, Y, X), not {}".format(data_shape)
            )

        self.sub_writer.create_position(position, pos_name)
        self.sub_writer.init_array(
            data_shape, chunk_size, dtype, chan_names, clims, overwrite
        )

    def write(self, data, p, t=None, c=None, z=None):
        """
        Wrapper that calls the builder's write function.
        Will write to existing array of zeros and place
        data over the specified indicies.

        :param np.array data: Data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param int p: Position index in which to write the data into
        :param int/slice t: Time index or index range of the time dimension
        :param int/slice c: Channel index or index range of the channel dimension
        :param int/slice z: Slice index or index range of the Z-slice dimension
        """
        self.sub_writer.open_position(p)

        if t is None:
            t = slice(0, data.shape[0])

        if c is None:
            c = slice(0, data.shape[1])

        if z is None:
            z = slice(0, data.shape[2])

        self.sub_writer.write(data, t, c, z)


class DefaultZarr(WriterBase):
    """
    This writer is based off creating a default HCS hierarchy for non-hcs datasets.
    Currently, we decide that all positions will live under individual columns under
    a single row.  i.e. this produces the following structure:
    Dataset.zarr
        ____> Row_0
            ---> Col_0
                ---> Pos_000
            ...
            --> Col_N
                ---> Pos_N
    We assume this structure in the metadata updating/position creation
    """

    def __init__(self, store, root_path):

        super().__init__(store, root_path)

        self.dataset_name = None
        self.plate_meta = dict()
        self.well_meta = dict()

    def init_hierarchy(self):
        """
        method to init the default hierarchy.
        Will create the first row and initialize metadata fields
        """
        self.create_row(0)
        self.dataset_name = os.path.basename(self.root_path).strip(".zarr")

        self.plate_meta["plate"] = {
            "acquisitions": [
                {"id": 1, "maximumfieldcount": 1, "name": "Dataset", "starttime": 0}
            ],
            "columns": [],
            "field_count": 1,
            "name": self.dataset_name,
            "rows": [],
            "version": "0.1",
            "wells": [],
        }

        self.plate_meta["plate"]["rows"].append({"name": self.rows[0]})

        self.well_meta["well"] = {"images": [], "version": "0.1"}
        self.well_meta = dict(self.well_meta)

    def create_position(self, position, name):
        """
        Creates a column and position subgroup given the index and name.  Name is
        provided by the main writer class

        :param int position: Index of the position to create
        :param str name: Name of the position subgroup
        """
        # get row name and create a column
        row_name = self.rows[0]
        self.create_column(0, position)
        col_name = self.columns[position]

        if self.verbose:
            print(f"Creating and opening subgroup {row_name}/{col_name}/{name}")

        # create position subgroup
        self.store[row_name][col_name].create_group(name)

        # update trackers
        self.current_pos_group = self.store[row_name][col_name][name]
        self.current_well_group = self.store[row_name][col_name]
        self.current_position = position

        # update ome-metadata
        self.positions[position] = {"name": name, "row": row_name, "col": col_name}
        self._update_plate_meta(position)
        self._update_well_meta(position)

    def _update_plate_meta(self, pos):
        """
        Updates the plate metadata which lives at the highest level (top store).
        This metadata carries information on the rows/columns and their paths.

        :param int pos: Position index to update the metadata
        """
        self.plate_meta["plate"]["columns"].append({"name": self.columns[pos]})
        self.plate_meta["plate"]["wells"].append(
            {"path": f"{self.rows[0]}/{self.columns[pos]}"}
        )
        self.store.attrs.put(self.plate_meta)

    def _update_well_meta(self, pos):
        """
        Updates the well metadata which lives at the column level.
        This metadata carries information about the positions underneath.
        Assumes only one position will ever be underneath this level.

        :param intpos: Index of the position to update
        """
        self.well_meta["well"]["images"] = [{"path": self.positions[pos]["name"]}]
        self.store[self.rows[0]][self.columns[pos]].attrs.put(self.well_meta)


class HCSZarrModifier(ZarrReader):
    """
    Interacts with an HCS zarr store to provide abstract array writing and metadata
    mutation.

    Warning: Although this class inherits ZarrReader, it is NOT read only, and therefore
    can make dangerous writes to the zarr store.

    :param str zarr_file: root zarr file containing the desired dataset
                            Note: assumes OME-NGFF HCS format compatibility
    :param bool enable_overwrite: whether to allow overwriting of already written data
    """

    def __init__(self, zarr_file, enable_creation=True, overwrite_ok=True):
        super().__init__(zarrfile=zarr_file)

        if not os.path.isdir(zarr_file):
            raise ValueError("file does not exist")
        try:
            creation = "a" if enable_creation else "r+"
            self.store = zarr.open(zarr_file, mode=creation)
        except:
            raise FileNotFoundError(
                "Path: {} is not a valid zarr store".format(zarr_file)
            )

        self.overwrite_ok = overwrite_ok

    def get_position_group(self, position):
        """
        Return the group at this position.

        :param int position: position of the group requested
        :return zarr.hierarchy.group group: group at this position
        """
        pos_info = self.position_map[position]
        well = pos_info["well"]
        pos = pos_info["name"]
        return self.store[well][pos]

    def get_position_meta(self, position):
        """
        Return the well-level metadata at at this position from .zattrs

        :param int position: position to retrieve metadata from
        :return dict metadata: .zattrs metadata at this position
        """
        position_group = self.get_position_group(position)
        return position_group.attrs.asdict()

    def init_untracked_array(self, data_array, position, name):
        """
        Write an array to a given position without updating the HCS metadata.
        Array will be stored under the given name, with chunk sizes equal to
        one x-y slice (where xy is assumed to be the last dimension).

        :param np.ndarray data_array: data of array.
        :param int position: position to write array to
        :param str name: name of array in storage
        """
        store = self.get_position_group(position=position)

        chunk_size = [1] * len(data_array.shape[:-2]) + list(data_array.shape[-2:])
        store.array(
            name=name,
            data=data_array,
            shape=data_array.shape,
            chunks=chunk_size,
            overwrite=self.overwrite_ok,
        )

    def get_untracked_array(self, position, name):
        """
        Gets the untracked array with name 'name' at the given position

        :param int position: Position index
        :return np.array pos: Array of name 'name', size can vary
        """
        pos = self.get_position_group(position)
        return pos[name][:]

    def write_meta_field(self, position, metadata, field_name):
        """
        Writes 'metadata' to position's well-level .zattrs metadata by either
        creating a new field according to 'metadata', or concatenating the
        given metadata to an existing field if found.

        Assumes that the zarr store group given follows the OMG-NGFF HCS
        format as specified here:
                https://ngff.openmicroscopy.org/latest/#hcs-layout

        Warning: Dangerous. Writing metadata fields above the image-level of
                an HCS hierarchy can break HCS compatibility

        :param zarr.heirarchy.grp zarr_group: zarr heirarchy group whose
                                            attributes to mutate
        :param dict metadata: metadata dictionary to write to JSON .zattrs
        :param str field_name: name of new/existing field to write to
        """
        store = self.get_position_group(position=position)

        assert isinstance(store, zarr.hierarchy.Group), (
            f"current_pos_group of type {type(store)}" " must be zarr.heirarchy.group"
        )

        # get existing metadata
        current_metadata = store.attrs.asdict()

        assert "multiscales" in current_metadata, (
            "Current position must reference"
            "image-level store and have metadata currently tracking images"
        )

        # alter depending on whether field is new or existing
        if field_name in current_metadata:
            current_metadata[field_name].update(metadata)
        else:
            current_metadata[field_name] = metadata

        store.attrs.update(current_metadata)
