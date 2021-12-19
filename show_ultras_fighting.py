#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import contextlib
import os
import random
import termios
import tty

import bwgame

import unittypes


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


UNIT_CHARS = {
    bwgame.UnitTypes(index): char
    for index, (_, char, _) in enumerate(unittypes.UNITTYPES)
}

UNIT_COLORS = {
    bwgame.UnitTypes(index): color
    for index, (_, _, color) in enumerate(unittypes.UNITTYPES)
}

COLOR2INT = {
    "black": 0,
    "red": 1,
    "green": 2,
    "brown": 3,  # on IBM, low-intensity yellow is brown
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "gray": 7,  # low-intensity white
    "no_color": 8,
    "orange": 9,
    "bright_green": 10,
    "yellow": 11,
    "bright_blue": 12,
    "bright_magenta": 13,
    "bright_cyan": 14,
    "white": 15,
}


def tty_render(chars, colors):
    """Returns chars as string with ANSI escape sequences."""
    rows, cols = len(chars), len(chars[0])
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            if not colors[i][j]:
                result += chars[i][j]
                continue
            result += "\033[%d;3%dm%s\033[0m" % (
                # & 8 checks for brightness.
                bool(colors[i][j] & 8),
                colors[i][j] & ~8,
                chars[i][j],
            )
    return result + "\033[0m"


def print_scene(funcs, go_back=True):
    rect = funcs.map_bounds()

    colno = rect.to.x // 32
    rowno = rect.to.y // 32

    chars = [[" " for _ in range(colno)] for _ in range(rowno)]
    colors = [[0 for _ in range(colno)] for _ in range(rowno)]

    for unit in funcs.st.visible_units():
        char = UNIT_CHARS.get(unit.unit_type.id)
        if char is None:
            print("unknown", unit)
            continue
        chars[unit.position.y // 32][unit.position.x // 32] = char

        color = UNIT_COLORS.get(unit.unit_type.id)
        if color == "default" or color is None:
            continue
        colors[unit.position.y // 32][unit.position.x // 32] = COLOR2INT[color]

    print("-" * colno)
    print(tty_render(chars, colors))
    framecount = "Frame %i " % funcs.st.current_frame
    print("%s%s" % (framecount, "-" * (colno - len(framecount))))

    if go_back:
        print("\033[%dA" % (rowno + 2))


def test(funcs, aensnared, bensnared):
    awins = bwins = 0
    pos_a = funcs.map_bounds().to // 2 - bwgame.XY(0, 10)
    pos_b = funcs.map_bounds().to // 2 + bwgame.XY(0, 10)

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

    for t in range(1000):
        funcs.next_frame()

        if t % 100 == 0:
            print_scene(funcs)

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

    return awins, bwins


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

    test(funcs, False, False)


if __name__ == "__main__":
    main()
