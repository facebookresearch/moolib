#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import contextlib
import random
import termios
import tty
import os

import bwgame


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


UNIT_CHARS = {
    bwgame.UnitTypes.Zerg_Ultralisk: "U",
    bwgame.UnitTypes.Special_Power_Generator: "G",
}


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

    def print_scene(go_back=True):
        rect = funcs.map_bounds()

        colno = rect.to.x // 32
        rowno = rect.to.y // 32

        scene = [[" " for _ in range(colno)] for _ in range(rowno)]

        for unit in st.visible_units():
            try:
                char = UNIT_CHARS[unit.unit_type.id]
            except ValueError:
                print("unknown", unit)
                continue
            scene[unit.position.y // 32][unit.position.x // 32] = char

        for row in scene:
            print("".join(row))
        print("-" * colno)

        if go_back:
            print("\033[%dA" % (rowno + 1))

    def test(aensnared, bensnared):
        awins = bwins = 0
        pos_a = funcs.map_bounds().to // 2 - bwgame.XY(0, 10)
        pos_b = funcs.map_bounds().to // 2 + bwgame.XY(0, 10)

        for i in range(1):
            a = funcs.trigger_create_unit(
                funcs.get_unit_type(bwgame.UnitTypes.Zerg_Ultralisk), pos_a, 0
            )
            b = funcs.trigger_create_unit(
                funcs.get_unit_type(bwgame.UnitTypes.Zerg_Ultralisk), pos_b, 1
            )
            if a is None or b is None:
                raise "Failed to spawn"

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

            for t in range(1000):
                funcs.next_frame()

                if t % 100 == 0:
                    print_scene()

                    with no_echo():
                        ch = ord(os.read(0, 1))
                    if ch == ord("q"):
                        return

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

    test(False, True)


if __name__ == "__main__":
    main()
