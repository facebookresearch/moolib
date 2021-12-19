#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import contextlib
import curses
import random

import bwgame

import unittypes


@contextlib.contextmanager
def scr():
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()  # No input buffering.
    curses.start_color()
    try:
        yield stdscr
    finally:
        stdscr.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()


UNIT_CHARS = {
    bwgame.UnitTypes(index): char
    for index, (_, char, _) in enumerate(unittypes.UNITTYPES)
}

UNIT_COLORS = {
    bwgame.UnitTypes(index): color
    for index, (_, _, color) in enumerate(unittypes.UNITTYPES)
}


def print_scene(funcs, win):
    win.erase()

    for unit in funcs.st.visible_units():
        char = UNIT_CHARS.get(unit.unit_type.id)
        if char is None:
            print("unknown", unit)
            continue

        win.insch(unit.position.y // 32, unit.position.x // 32, char)

        color = UNIT_COLORS.get(unit.unit_type.id)
        if color == "default" or color is None:
            continue
        del color  # TODO: Use.


def ultratest(funcs, aensnared, bensnared):
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

        if t % 50 == 0:
            yield

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

    with scr() as stdscr:
        mapcols, maprows = funcs.map_bounds().to // 32
        pad = curses.newpad(maprows, mapcols)
        pad.keypad(True)  # Get curses.KEY_LEFT etc.
        pminrow, pmincol = maprows // 2, mapcols // 4  # Starting pos of pad.

        scrrows, scrcols = stdscr.getmaxyx()

        botl = curses.newwin(1, scrcols, scrrows - 1, 0)

        for _ in ultratest(funcs, False, False):
            print_scene(funcs, pad)

            botl.addstr(0, 0, "Frame %i" % funcs.st.current_frame, curses.A_REVERSE)
            botl.noutrefresh()

            unit = funcs.get_first_selected_unit(0)
            if unit is not None:
                pad.move(*reversed(unit.position // 32))
            else:
                pad.move(0, 0)
            pad.noutrefresh(pminrow, pmincol, 0, 0, scrrows - 2, scrcols - 1)

            curses.doupdate()
            ch = pad.getch()
            if ch == ord("q"):
                return
            elif ch == curses.KEY_DOWN:
                pminrow += 1
                print("down")
            elif ch == curses.KEY_UP:
                pminrow -= 1
            elif ch == curses.KEY_LEFT:
                pmincol -= 10
            elif ch == curses.KEY_RIGHT:
                pmincol += 10


if __name__ == "__main__":
    main()
