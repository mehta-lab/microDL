import iohub.ngff as ngff
import zarr


def write_untracked_array(
    position: ngff.Position, data_array, name, overwrite_ok=False
):
    """
    Write an array to a given position without updating the HCS omero/multiscales
    metadata.

    Array will be stored under the given name, with chunk sizes equal to
    one x-y slice (where xy is assumed to be the last dimension).

    :param Position zarr_dir: NGFF position node object
    :param np.ndarray data_array: data of array
    :param str name: name of array in storage
    :param bool overwrite_ok: whether overwriting existing arrays of same name is
                    allowed (generally for flatfielding), by default False
    """
    chunk_size = [1] * len(data_array.shape[:-2]) + list(data_array.shape[-2:])
    position.zgroup.array(
        name=name,
        data=data_array,
        shape=data_array.shape,
        chunks=chunk_size,
        overwrite=overwrite_ok,
    )


def write_meta_field(position: ngff.Position, metadata, field_name):
    """
    Writes 'metadata' to position's well-level .zattrs metadata by either
    creating a new field according to 'metadata', or concatenating the
    given metadata to an existing field if found.

    Assumes that the zarr store group given follows the OMG-NGFF HCS
    format as specified here:
            https://ngff.openmicroscopy.org/latest/#hcs-layout

    Warning: Dangerous. Writing metadata fields above the image-level of
            an HCS hierarchy can break HCS compatibility

    :param Position zarr_dir: NGFF position node object
    :param dict metadata: metadata dictionary to write to JSON .zattrs
    :param str field_name: name of new/existing field to write to
    """
    if field_name in position.zattrs:
        position.zattrs[field_name].update(metadata)
    else:
        position.zattrs[field_name] = metadata


def get_untracked_array_slice(
    position: ngff.Position, meta_field_name, time_index, channel_index, z_index
):
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
        meta_field = position.zattrs[meta_field_name]
        array_name = meta_field["array_name"]
        channels = meta_field["channel_ids"]
    except Exception as e:
        raise AttributeError(
            f"Caught {e}. No metadata field found for '{meta_field_name}'."
        )
    # untracked array might have collapsed indices
    channel_pos = channels.index(channel_index)
    return position.zgroup[array_name][time_index, channel_pos, z_index]


def add_channel(
    position: ngff.Position,
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
    assert len(new_channel_array.shape) == len(position.data.shape) - 1, (
        "New channel array must match all dimensions of the position array, "
        "except in the inferred channel dimension: "
        f"array shape: {position.data.shape}"
        f", expected channel shape: {(position.data.shape[0], ) + position.data.shape[2:]}"
        f", received channel shape: {new_channel_array.shape}"
    )
    # determine whether to overwrite or append
    if new_channel_name in position.channel_names and overwrite_ok:
        new_channel_index = list(position.channel_names).index(new_channel_name)
    else:
        new_channel_index = len(position.channel_names)
        position.append_channel(new_channel_name, resize_arrays=True)

    # replace or append channel
    position["0"][:, new_channel_index] = new_channel_array
