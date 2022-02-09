
# moolib examples

## Simplified example agent

To try out the simplified A2C example locally, run

```
python examples/a2c.py
```

This produces a file called `logs.tsv` which can be plotted using
`plot.py`. To see results during training, you can run

```
watch -n3 --color python examples/plot.py logs.tsv --ykey mean_episode_return --window 500
```

in another terminal.

Here's a sample result:

```
                                mean_episode_return
  180 +---------------------------------------------------------------------+
      |           +          +           +    AAAAAAAAAAAA      +           |
  160 |-+.........:..........:..........AAAAAAA......:..........:.........+-|
      |           :          :    AAAAAAA:           :          :           |
      |           :          AAAAAA      :           :          :           |
  140 |-+.........:........AAA...........:...........:..........:.........+-|
      |           :      AAA :           :           :          :           |
  120 |-+.........AAAAAAAA...:...........:...........:..........:.........+-|
      |        AAAAA         :           :           :          :           |
      |       AA  :          :           :           :          :           |
  100 |-+....AA...:..........:...........:...........:..........:.........+-|
      |    AAA    :          :           :           :          :           |
   80 |-+.AA......:..........:...........:...........:..........:.........+-|
      |  A        :          :           :           :          :           |
      | AA        :          :           :           :          :           |
   60 |-A.........:..........:...........:...........:..........:.........+-|
      | A         :          :           :           :          :           |
   40 |AA.........:..........:...........:...........:..........:.........+-|
      |A          :          :           :           :          :           |
      |A          :          :           :           :          :           |
   20 |-+.........:..........:...........:...........:..........:.........+-|
      |           +          +           +           +          +           |
    0 +---------------------------------------------------------------------+
      0         50000      100000      150000      200000     250000      300000
                                       step
                                 logs.tsv +--A--+
```


## Fully-fledged vtrace agent

### Running

To run the example agent on a given Atari level:

First, start the broker:

    python -m moolib.broker

It will output something like `Broker listening at 0.0.0.0:4431`.

Note that a **single broker is enough** for all your experiments.

Now take the IP address of your computer. If you ssh'd into your
machine, this should work (in a new shell):

```
export BROKER_IP=$(echo $SSH_CONNECTION | cut -d' ' -f3)  # Should give your machine's IP.
export BROKER_PORT=4431
```

To start an experiment with a single peer:

    python -m examples.vtrace.experiment connect=BROKER_IP:BROKER_PORT \
        savedir=/tmp/moolib-atari/savedir \
        project=moolib-atari \
        group=Zaxxon-Breakout \
        env.name=ALE/Breakout-v5

To add more peers to this experiment, start more processes with the
same `project` and `group` settings, using a different setting for
`device` (default: `'cuda:0'`) if on the same machine.


### Batch sizes in example agent.

In the `moolib` example agent(s), there are several different batch sizes:

  * The `actor_batch_size`, i.e., the second dimension of the model
    inputs at acting time: The `B` in `[1, B, W, H, C]`.
    (2x actor batch size is the number of environment instances due to
    'double buffering')

  * The learner batch size (often just `batch_size`), i.e. the `B` in
    `[T, B, W, H, C]` at learning time (to produce local gradients).

  * The unroll length is a batch size of sorts, i.e. the `T` in `[T,
    B, W, H, C]` at learning time. Only when using RNNs (agents with
    memory) is this partially treated as a sequence length as well.

  * The virtual batch size, i.e., the number of samples in the `B`
    dimension moolib consumes and adds to its gradient buffers before
    a gradient descent step is happening. This can happen in two ways:
    (1) A single peer could go through several backprop steps and keep
    adding to its "running gradient" before it applies the gradient,
    or (2) multiple peers go through one sample each and accumulate
    their gradients. The virtual batch size is then
    `number_of_peers * learner_batch_size`.

Note that the `virtual_batch_size` setting in moolib is (currently) a
_lower bound_ on the number of samples required to do a single grad
descent step. When using several peers in parallel, `moolib` can
overshoot. Logging the `gradient_stats["batch_size"]` entry tells you
what the actual virtual batch size has been at each step. The reason
`moolib` treats the `accumulator.set_virtual_batch_size` value as a lower
bound (instead of as an lower and upper bound) is that it would
otherwise need to do more synchronisation, which would reduce overall
throughput.
