# Copyright (c) Facebook, Inc. and its affiliates.
#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import random

import bwgame


def test(funcs, aensnared, bensnared):
    awins = bwins = 0
    pos_a = funcs.map_bounds().to // 2 - bwgame.XY(0, 10)
    pos_b = funcs.map_bounds().to // 2 + bwgame.XY(0, 10)

    for _ in range(100):
        a = funcs.trigger_create_unit(
            funcs.get_unit_type(bwgame.UnitTypes.Zerg_Ultralisk), pos_a, 0
        )
        b = funcs.trigger_create_unit(
            funcs.get_unit_type(bwgame.UnitTypes.Zerg_Ultralisk), pos_b, 1
        )
        if a is None or b is None:
            raise RuntimeError("Failed to spawn")

        a.heading = funcs.xy_direction(b.position - a.position)
        b.heading = funcs.xy_direction(a.position - b.position)

        funcs.action_select(0, a)
        funcs.action_order(
            0,
            funcs.get_order_type(bwgame.Orders.AttackDefault),
            bwgame.XY(),
            b,
            None,
            False,
        )

        funcs.action_select(1, b)
        funcs.action_order(
            1,
            funcs.get_order_type(bwgame.Orders.AttackDefault),
            bwgame.XY(),
            a,
            None,
            False,
        )

        if aensnared:
            funcs.ensnare_unit(a)
        if bensnared:
            funcs.ensnare_unit(b)

        for _ in range(1000):
            funcs.next_frame()
            if funcs.unit_dead(a) or funcs.unit_dead(b):
                break

        if not funcs.unit_dead(a):
            awins += 1

        if not funcs.unit_dead(b):
            bwins += 1

        if not funcs.unit_dead(a):
            funcs.hide_unit(a)
            funcs.kill_unit(a)

        if not funcs.unit_dead(b):
            funcs.hide_unit(b)
            funcs.kill_unit(b)

    return awins, bwins


def measure(title, f):
    results = []

    for _ in range(10):
        aw, bw = f()
        winrate = aw / (aw + bw)
        results.append(winrate)
    results.sort()

    L = len(results)

    avg = sum(results) / L
    variance = sum((v - avg) ** 2 for v in results) / L

    print("\n-- %s --" % title)
    print("min: %g" % min(results))
    print("max: %g" % max(results))
    print("avg: %g" % avg)
    print(
        "med: %g  [%g %g %g %g %g]"
        % (
            results[L // 2],
            results[L // 6],
            results[L // 3],
            results[L // 3 + L // 6],
            results[-L // 3],
            results[-L // 6],
        )
    )
    print("stddev: %g, variance: %g" % (variance ** 0.5, variance))


def main():
    player = bwgame.GamePlayer(".")
    st = player.st()

    def setup_f():
        game_funcs.setup_info.starting_units = True
        game_funcs.setup_info.create_no_units = True

        st.players[0] = bwgame.Player.controller_occupied
        st.players[1] = bwgame.Player.controller_occupied

        st.lcg_rand_state = random.getrandbits(32)

    game_funcs = bwgame.GameLoadFunctions(st)
    game_funcs.load_map_file("maps/BroodWar/AIIDE/(2)Benzene.scx", setup_f)

    action_st = bwgame.ActionState()
    funcs = bwgame.ActionFunctions(st, action_st)

    measure("plain", lambda: test(funcs, False, False))
    measure("a ensnared", lambda: test(funcs, True, False))
    measure("b ensnared", lambda: test(funcs, False, True))
    measure("both ensnared", lambda: test(funcs, True, True))


if __name__ == "__main__":
    main()
