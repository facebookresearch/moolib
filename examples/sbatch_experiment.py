#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Runs a single experiment (multiple moolib peers) using sbatch.
#
# Test with python -m scripts/sbatch_experiment --dry
#
# Run w/o --dry.
#
import argparse
import ctypes
import getpass
import os
import socket
import sys

import coolname
import moolib

DEFAULT_PORT = 4431

parser = argparse.ArgumentParser(
    description="Training with slurm",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--project",
    default="moolib-atari",
    type=str,
    help="Project name.",
)
parser.add_argument(
    "--group",
    default=coolname.generate_slug(2),
    type=str,
    help="Group name. Defaults to a coolname slug.",
)
parser.add_argument("--dry", action="store_true")
parser.add_argument(
    "-n",
    "--num_peers",
    default=1,
    type=int,
    metavar="N",
    help="Number of peers (jobs in array) in this experiment.",
)
parser.add_argument(
    "--time",
    default=60 * 24,
    type=int,
    metavar="T",
    help="Maximum time this experiment runs (in min).",
)
parser.add_argument(
    "--constraint",
    default="",  # eg: "bldg2,volta32gb"
    type=str,
    metavar="constr",
    help="Matching this job constraint.",
)
parser.add_argument(
    "--partition",
    default="learnlab,learnfair",
    type=str,
    metavar="part",
    help="Request a specific partition for the resource allocation.",
)
parser.add_argument(
    "--cmd",
    default=(
        "python -m examples.appo.experiment connect=%(broker)s savedir=%(savedir)s"
        " project=%(project)s group=%(group)s"
    ),
    type=str,
    metavar="cmd",
    help="The command to run.",
)
parser.add_argument(
    "--broker",
    default="",
    type=str,
    metavar="addr:port",
    help="The address of the broker.",
)
parser.add_argument(
    "args",
    nargs="*",
    default=[],
    help="Extra arguments.",
    type=str,
)
parser.add_argument("--no-checks", action="store_true", help="Don't run checks.")


def get_address(address):
    if address:
        return address

    ssh_connection = os.getenv("SSH_CONNECTION")
    if ssh_connection:
        try:
            client_ip, client_port, host_ip, host_port = ssh_connection.split()
            if len(host_ip.split(".")) == 4:
                return "%s:%i" % (host_ip, DEFAULT_PORT)
        except ValueError:
            pass

    return "%s:%i" % (socket.gethostbyname(socket.gethostname()), DEFAULT_PORT)


def check_nfs(nfs_super_magic=0x6969):
    try:
        # See statfs(2).
        libc = ctypes.CDLL("libc.so.6")
        Statfs = ctypes.c_uint * 32
        buf = Statfs()
        ret = libc.statfs(".", buf)
        if ret != 0:
            return
        if buf[0] != nfs_super_magic:
            raise RuntimeError(
                "Must run from NFS directory, but cwd (%s) isn't (0x%x)"
                % (os.getcwd(), buf[0]),
            )
    except OSError:
        pass


def check_broker_online(address):
    rpc = moolib.Rpc()
    rpc.connect(address)
    rpc.set_timeout(2)
    try:
        rpc.sync("broker", "")
    except RuntimeError as e:
        # TODO: Add "ping" feature to moolib so we don't need to do _this_!
        if "timed out" in str(e):
            raise


def check(address):
    if FLAGS.no_checks:
        return

    try:
        check_broker_online(address)
    except RuntimeError as e:
        print("Couldn't reach broker at %s. Is it online? (Error: %s)" % (address, e))
        sys.exit(1)

    try:
        check_nfs()
    except RuntimeError as e:
        print(str(e))
        sys.exit(2)


def cmdlist(args):
    return ["sbatch"] + ["%s=%s" % item for item in args.items()]


def main():
    global FLAGS
    FLAGS = parser.parse_args()

    address = get_address(FLAGS.broker)
    check(address)

    savedir = os.path.join("/checkpoint", getpass.getuser(), FLAGS.project, FLAGS.group)

    try:
        os.makedirs(savedir)
    except FileExistsError:
        sys.stderr.write("Warning: Savedir path '%s' already exists\n" % savedir)

    if not os.access(savedir, os.W_OK | os.X_OK):
        sys.stderr.write("No write access to '%s'\n" % savedir)
        sys.exit(1)

    cmd = FLAGS.cmd % {
        "savedir": savedir,
        "broker": address,
        "project": FLAGS.project,
        "group": FLAGS.group,
    }

    slurm_output = os.path.join(savedir, "slurm-%A_%a.out")

    args = {
        "--constraint": FLAGS.constraint,
        "--job-name": "%s/%s" % (FLAGS.project, FLAGS.group),
        "--array": "0-%i" % (FLAGS.num_peers - 1),
        "--partition": FLAGS.partition,
        "--cpus-per-task": 10,
        "--gpus-per-task": 1,
        "--mem-per-cpu": "8G",
        "--time": FLAGS.time,
        "--ntasks": 1,
        "--output": slurm_output,
        "--error": slurm_output,
        "--export": "ALL",
        "--wrap": " ".join(cmd.split() + FLAGS.args),
    }

    execvlist = cmdlist(args)

    # Can't extra-escape strings for execvp, but want that for printing.
    args["--wrap"] = repr(args["--wrap"])
    print(" ".join(cmdlist(args)))

    if FLAGS.dry:
        return

    os.execvp("sbatch", execvlist)


if __name__ == "__main__":
    main()
