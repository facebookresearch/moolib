#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import collections
import contextlib
import curses
import enum

import bwgame

import unittypes


class Color(enum.IntEnum):
    BLACK = 0
    RED = 1
    GREEN = 2
    BROWN = 3  # On IBM, low-intensity yellow is brown.
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    GRAY = 7  # Low-intensity white.
    NO_COLOR = 8
    ORANGE = 9
    BRIGHT_GREEN = 10
    YELLOW = 11
    BRIGHT_BLUE = 12
    BRIGHT_MAGENTA = 13
    BRIGHT_CYAN = 14
    WHITE = 15


def init_colors():
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Actually brown.
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_CYAN, -1)
    curses.init_pair(8, curses.COLOR_WHITE, -1)  # Gray.
    curses.init_pair(9, -1, -1)

    # "Bright black" is dark gray.
    colors = [curses.color_pair(Color.BLACK + 1) | curses.A_BOLD]

    # Low-intensity colors.
    for c in range(Color.RED, Color.ORANGE):
        colors.append(curses.color_pair(c + 1))

    # Bright versions.
    for c in range(Color.RED, len(Color)):
        colors.append(colors[c] | curses.A_BOLD)

    return colors


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


SYMSET = {
    bwgame.UnitTypes(index): (char, color)
    for index, (_, char, color) in enumerate(unittypes.UNITTYPES)
}


def print_scene(g):
    g.pad.erase()

    for unit in g.funcs.st.visible_units():
        char, color = SYMSET.get(unit.unit_type.id, (None, None))
        if char is None:
            print("unknown", unit)
            continue
        if color == "default" or color is None:
            color = "no_color"
        color = g.colors[Color[color.upper()]]
        g.pad.insch(unit.position.y // 32, unit.position.x // 32, char, color)


def replay(funcs):
    for t in range(1000):
        if funcs.is_done():
            break

        funcs.next_frame()

        if t % 50 == 0:
            yield


InstanceGlobals = collections.namedtuple(
    "InstanceGlobals", ["funcs", "pad", "botl", "colors", "scrrows", "scrcols"]
)


def main():
    player = bwgame.GamePlayer("")
    st = player.st()

    action_st = bwgame.ActionState()
    replay_st = bwgame.ReplayState()
    funcs = bwgame.ReplayFunctions(st, action_st, replay_st)
    funcs.load_replay_file("maps/p49.rep")

    units = set()
    with scr() as stdscr:
        colors = init_colors()
        mapcols, maprows = funcs.map_bounds().to // 32
        pad = curses.newpad(maprows, mapcols)
        pad.keypad(True)  # Get curses.KEY_LEFT etc.
        curses.mousemask(-1)
        pminrow, pmincol = maprows // 2, mapcols // 4  # Starting pos of pad.

        scrrows, scrcols = stdscr.getmaxyx()

        botl = curses.newwin(1, scrcols, scrrows - 1, 0)

        g = InstanceGlobals(
            funcs=funcs,
            pad=pad,
            botl=botl,
            colors=colors,
            scrrows=scrrows,
            scrcols=scrcols,
        )

        while not funcs.is_done():
            print_scene(g)
            funcs.next_frame()

            if st.current_frame % 50 != 0:
                continue

            for unit in g.funcs.st.visible_units():
                units.add(unit.unit_type.id)

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
                break
            elif ch == (ord("j"), curses.KEY_DOWN):
                pminrow += 1
            elif ch in (ord("k"), curses.KEY_UP):
                pminrow -= 1
            elif ch in (ord("h"), curses.KEY_LEFT):
                pmincol -= 10
            elif ch in (ord("l"), curses.KEY_RIGHT):
                pmincol += 10
            elif ch == curses.KEY_RESIZE:
                scrrows, scrcols = stdscr.getmaxyx()
                botl.mvwin(scrrows - 1, 0)
            elif ch == ord("\t"):
                pass  # Tab.
            elif ch == curses.KEY_MOUSE:
                print(curses.getmouse())

    print("Thanks for playing.")
    print(units)


if __name__ == "__main__":
    main()
