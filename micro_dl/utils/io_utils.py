import iohub.ngff as ngff
import zarr

def init_untracked_array(zarr_dir, position_path, data_array, name, overwrite_ok=False):
    """
    Write an array to a given position without updating the HCS omero/multiscales
    metadata.
    
    Array will be stored under the given name, with chunk sizes equal to
    one x-y slice (where xy is assumed to be the last dimension).

    :param str zarr_dir: path to zarr directory
    :param np.ndarray data_array: data of array.
    :param int position_path: relative path in store to position to write array to
    :param str name: name of array in storage
    :param bool overwrite_ok: whether overwriting existing arrays of same name is
                    allowed (generally for flatfielding), by default False
    """

    plate = zarr.open(zarr_dir, mode="r+")
    store = plate[position_path]
    
    #write the array
    chunk_size = [1] * len(data_array.shape[:-2]) + list(data_array.shape[-2:])
    store.array(
        name=name,
        data=data_array,
        shape=data_array.shape,
        chunks=chunk_size,
        overwrite=overwrite_ok,
    )

def write_meta_field(zarr_dir, position_path, metadata, field_name):
    """
    Writes 'metadata' to position's well-level .zattrs metadata by either
    creating a new field according to 'metadata', or concatenating the
    given metadata to an existing field if found.

    Assumes that the zarr store group given follows the OMG-NGFF HCS
    format as specified here:
            https://ngff.openmicroscopy.org/latest/#hcs-layout

    Warning: Dangerous. Writing metadata fields above the image-level of
            an HCS hierarchy can break HCS compatibility

    :param str zarr_dir: path to zarr directory
    :param int position_path: relative path in store to position to write array to
    :param dict metadata: metadata dictionary to write to JSON .zattrs
    :param str field_name: name of new/existing field to write to
    """
    plate = zarr.open(zarr_dir, mode="r+")
    store = plate[position_path]

    assert isinstance(store, zarr.hierarchy.Group), (
        f"current_pos_group of type {type(store)}" " must be zarr.heirarchy.group"
    )

    # get existing metadata
    current_metadata = store.attrs.asdict()

    assert "multiscales" in current_metadata, (
        "Current position must reference "
        "position-level store and have metadata currently tracking images"
    )

    # alter depending on whether field is new or existing
    if field_name in current_metadata:
        current_metadata[field_name].update(metadata)
    else:
        current_metadata[field_name] = metadata

    store.attrs.update(current_metadata)

def read_meta_field(zarr_dir, position_path, field_name):
    """
    Reads specified metadata field from the .zattrs metadata of the specified
    position

    Assumes that the zarr store group given follows the OMG-NGFF HCS
    format as specified here:
            https://ngff.openmicroscopy.org/latest/#hcs-layout

    :param str zarr_dir: path to zarr directory
    :param int position_path: relative path in store to position to write array to
    :param str field_name: name of field to read
    """
    plate = zarr.open(zarr_dir, mode="r+")
    store = plate[position_path]

    assert isinstance(store, zarr.hierarchy.Group), (
        f"current_pos_group of type {type(store)}" " must be zarr.heirarchy.group"
    )
    
    # get existing metadata
    current_metadata = store.attrs.asdict()
    assert "multiscales" in current_metadata, (
        "Current position must reference "
        "position-level store and have metadata currently tracking images"
    )
    
    return current_metadata[field_name]

def get_untracked_array_slice(zarr_dir, position_path, meta_field_name, time_index, channel_index, z_index):
    """
    Get z-slice of untracked (not in multiscales/omero) array given a position and a channel.
    
    :param str zarr_dir: path to zarr directory
    :param str position_path: position path to position inside store
    :param int time_index: time id to use for selecting slice
    :param int channel_index: channel id to use for selecting slice
    :param int z_index: z-stack depth id to use for selecting slice
    :param str meta_field_name: name of untracked array's metadata field

    :return np.ndarray slice: 2D slice as numpy array
    """

    try:
        position_metadata = read_meta_field(zarr_dir, position_path, meta_field_name)
        array_name = position_metadata["array_name"]
        channels = position_metadata["channel_ids"]
    except Exception as e:
        raise AttributeError(f"Caught {e}. No metadata field found for '{meta_field_name}'.")

    
    store = zarr.open(zarr_dir, 'r')
    array = store[position_path][array_name]

    # untracked array might have collapsed indices
    channel_pos = channels.index(channel_index)
    array_slice = array[time_index, channel_pos, z_index, :, :]

    return array_slice

def add_channel(
    zarr_dir,
    position_path,
    new_channel_array,
    new_channel_name,
    overwrite_ok=False,
):
    """
    Adds a channels to the data array at position "position". Note that there is
    only one 'tracked' data array in current HCS spec at each position. Also
    updates the 'omero' channel-tracking metadata to track the new channel.

    The 'new_channel_array' must match the dimensions of the current array in
    all positions but the channel position (1) and have the same datatype

    Note: to maintain HCS compatibility of the zarr store, all positions (wells)
    must maintain arrays with congruent channels. That is, if you add a channel
    to one position of an HCS compatible zarr store, an additional channel must
    be added to every position in that store to maintain HCS compatibility.

    :param str zarr_dir: path to zarr directory
    :param str position_path: path to position in zarr store 
    :param np.ndarray new_channel_array: array to add as new channel with matching
                            dimensions (except channel dim) and dtype
    :param str new_channel_name: name of new channel
    :param bool overwrite_ok: if true, if a channel with the same name as
                            'new_channel_name' is found, will overwrite
    """
    plate = ngff.open_ome_zarr(zarr_dir, mode="r+")
    position = plate[position_path]
    
    assert len(new_channel_array.shape) == len(position.data.shape) - 1, (
        "New channel array must match all dimensions of the position array, "
        "except in the inferred channel dimension: "
        f"array shape: {position.data.shape}"
        f", expected channel shape: {(position.data.shape[0], ) + position.data.shape[2:]}"
        f", received channel shape: {new_channel_array.shape}"
    )
    #determine whether to overwrite or append
    if new_channel_name in position.channel_names and overwrite_ok:
        new_channel_index = list(position.channel_names).index(new_channel_name)
    else:
        new_channel_index = len(position.channel_names)
        position.append_channel(new_channel_name, resize_arrays=True)
    
    # replace or append channel
    position["0"][:, new_channel_index] = new_channel_array
    
    plate.close()