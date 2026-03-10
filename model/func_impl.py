import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    is_fc1: bool,
    is_megatron_mp: bool,
    in_dim: int,
    out_dim: int,
):
    """The function that prepare necessary information for parallel training.

    Parameters
    ----------
        comm : Communicator
            the global mpi communicator

        rank : int
            the corresponding rank of the process

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        is_fc1 : int
            A boolean indicating whether the current layer is the first layer or not

        is_megatron_mp : boolean
            A boolean indicating whether we are using Megatron-style Model Parallel or not

        in_dim : int
            An integer corresponds to the original input feature dimension

        out_dim : int
            An integer corresponds to the original output feature dimension

    Returns
    -------
        mp_idx : int
            An integer corresponds to model parallel communication index

        dp_idx : int
            An integer corresponds to data parallel communication index

        mp_comm : Communicator
            The Model Parallel communicator after split

        dp_comm : Communicator
            The Data Parallel communicator after split

        part_in_dim : int
            An integer corresponds to the input feature dimension after specific parallelism

        part_out_dim : int
            An integer corresponds to the output feature dimension after specific parallelism
    """

    # MP-major layout:
    # rank = dp_idx * mp_size + mp_idx
    mp_idx = rank % mp_size
    dp_idx = rank // mp_size

    # MP groups keep dp_idx fixed; DP groups keep mp_idx fixed.
    mp_comm = comm.Split(key=mp_idx, color=dp_idx)
    dp_comm = comm.Split(key=dp_idx, color=mp_idx)

    # Naive MP partitions output dim for both FC layers.
    # Megatron MP partitions:
    #   - FC1 on output dim
    #   - FC2 on input dim
    if is_megatron_mp and (not is_fc1):
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim
    else:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with naive model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    gathered_x = np.empty((mp_size,) + x.shape, dtype=x.dtype)
    mp_comm.Allgather(x, gathered_x)
    return np.concatenate(np.split(gathered_x, mp_size, axis=0), axis=2).squeeze(axis=0)


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with naive model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    gathered_out = np.empty((mp_size,) + out.shape, dtype=out.dtype)
    mp_comm.Allgather(out, gathered_out)
    return np.concatenate(np.split(gathered_out, mp_size, axis=0), axis=2).squeeze(axis=0)


def megatron_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    return x


def megatron_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    collected_out = np.empty_like(out)
    mp_comm.Allreduce(out, collected_out, op=MPI.SUM)
    return collected_out


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with naive model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    return np.split(output_grad, mp_size, axis=1)[mp_group_idx]


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with naive model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """

    part_in_dim = grad_x.shape[1] // mp_size
    packed_grad_x = np.concatenate(
        [chunk.reshape(-1) for chunk in np.split(grad_x, mp_size, axis=1)],
        axis=0,
    )
    reduced_shard = np.empty(grad_x.shape[0] * part_in_dim, dtype=grad_x.dtype)
    mp_comm.Reduce_scatter(packed_grad_x, reduced_shard, op=MPI.SUM)
    return reduced_shard.reshape(grad_x.shape[0], part_in_dim)


def megatron_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with megatron-style model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    return output_grad


def megatron_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with megatron-style model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """

    return grad_x


def collect_weight_grad(
    grad_w: np.ndarray,
    grad_b: np.ndarray,
    dp_comm,
):
    """The function for collecting weight gradients across data parallel nodes

    Parameters
    ----------
        grad_w : np.ndarray
            gradients value for fc weight on a single node of shape (in_dim, out_dim)

        grad_b : np.ndarray
            gradients value for fc bias on a single node of shape (1, out_dim)

        dp_comm : Communicator
            The Data Parallel communicator

    Returns
    -------
        collected_grad_w : np.ndarray
            collected gradients value of shape (in_dim, out_dim) for fc weight across different nodes

        collected_grad_b : np.ndarray
            collected gradients value of shape (1, out_dim) for fc bias across different nodes

    """

    collected_grad_w = np.empty_like(grad_w)
    collected_grad_b = np.empty_like(grad_b)
    dp_comm.Allreduce(grad_w, collected_grad_w, op=MPI.SUM)
    dp_comm.Allreduce(grad_b, collected_grad_b, op=MPI.SUM)
    dp_group_size = dp_comm.Get_size()
    collected_grad_w /= dp_group_size
    collected_grad_b /= dp_group_size
    return collected_grad_w, collected_grad_b
