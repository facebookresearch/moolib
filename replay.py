#
# Inspired by C++ example at
#   https://gist.github.com/tscmoo/f10446517515828828b0a188ca3911a2
#

import contextlib
import curses
import dataclasses
import enum
import time

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

    @classmethod
    def get(cls, name):
        if name is None or name == "default":
            return cls.NO_COLOR
        return cls[name.upper()]


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


class ColorMode(enum.Enum):
    CUSTOM = 0
    RACE = 1
    PLAYER = 2
    FRIENDFOE = 3


PLAYERCOLORS = [
    Color.RED,
    Color.BRIGHT_BLUE,
    Color.CYAN,  # Teal.
    Color.MAGENTA,  # Purple.
    Color.ORANGE,
    Color.BROWN,  # Green on ice maps.
    Color.WHITE,
    Color.YELLOW,
]

RACECOLORS = {
    bwgame.Race.zerg: Color.MAGENTA,
    bwgame.Race.terran: Color.BRIGHT_BLUE,
    bwgame.Race.protoss: Color.YELLOW,
    bwgame.Race.none: Color.WHITE,
}


def print_scene(g):
    g.pad.erase()

    for unit in g.funcs.st.visible_units():
        char, color = SYMSET.get(unit.unit_type.id, (None, None))
        if char is None:
            print("unknown", unit)
            continue
        race = g.funcs.unit_race(unit)
        if g.colormode == ColorMode.CUSTOM or race == bwgame.Race.none:
            color = g.colors[Color.get(color)]
        elif g.colormode == ColorMode.RACE:
            color = g.colors[RACECOLORS[race]]
        elif g.colormode == ColorMode.PLAYER:
            color = g.colors[PLAYERCOLORS[unit.owner]]
        elif g.colormode == ColorMode.FRIENDFOE:
            for n in range(8):
                if g.funcs.player_slot_active(n):
                    break
            color = g.colors[Color.BRIGHT_GREEN if unit.owner == n else Color.ORANGE]
        if g.funcs.ut_building(unit.unit_type):
            color |= curses.A_REVERSE
        g.pad.insch(unit.position.y // 32, unit.position.x // 32, char, color)


def replay(funcs):
    for t in range(1000):
        if funcs.is_done():
            break

        funcs.next_frame()

        if t % 50 == 0:
            yield


@dataclasses.dataclass
class InstanceGlobals:
    funcs: bwgame.StateFunctions
    pad: curses.window
    botl: curses.window
    colors: list
    scrrows: int
    scrcols: int
    colormode: ColorMode


def main():
    player = bwgame.GamePlayer("")
    st = player.st()

    action_st = bwgame.ActionState()
    replay_st = bwgame.ReplayState()
    funcs = bwgame.ReplayFunctions(st, action_st, replay_st)
    funcs.load_replay_file("maps/p49.rep")

    with scr() as stdscr:
        colors = init_colors()
        mapcols, maprows = funcs.map_bounds().to // 32
        pad = curses.newpad(maprows, mapcols)
        pad.keypad(True)  # Get curses.KEY_LEFT etc.
        curses.mousemask(-1)

        scrrows, scrcols = stdscr.getmaxyx()
        # Starting pos of pad: Middle of map.
        pminrow = maprows // 2 - scrrows // 2
        pmincol = mapcols // 2 - scrcols // 2

        botl = curses.newwin(1, scrcols, scrrows - 1, 0)

        g = InstanceGlobals(
            funcs=funcs,
            pad=pad,
            botl=botl,
            colors=colors,
            scrrows=scrrows,
            scrcols=scrcols,
            colormode=ColorMode.PLAYER,
        )

        pad.nodelay(True)
        time_delta = 0.02

        while not funcs.is_done():
            print_scene(g)
            time.sleep(time_delta)
            funcs.next_frame()

            botl.addstr(0, 0, "Frame %i" % funcs.st.current_frame, curses.A_REVERSE)
            botl.noutrefresh()

            pad.noutrefresh(pminrow, pmincol, 0, 0, scrrows - 2, scrcols - 1)

            curses.doupdate()
            ch = pad.getch()
            if ch == -1:  # No input.
                continue
            elif ch == ord("q"):
                break
            elif ch in (ord("j"), curses.KEY_DOWN):
                pminrow += min(10, maprows - pminrow - scrrows + 1)
            elif ch in (ord("k"), curses.KEY_UP):
                pminrow -= min(10, pminrow)
            elif ch in (ord("h"), curses.KEY_LEFT):
                pmincol -= min(10, pmincol)
            elif ch in (ord("l"), curses.KEY_RIGHT):
                pmincol += min(10, mapcols - pmincol - scrcols + 1)
            elif ch == curses.KEY_RESIZE:
                scrrows, scrcols = stdscr.getmaxyx()
                botl.mvwin(scrrows - 1, 0)
            elif ch == ord("\t"):
                g.colormode = ColorMode((g.colormode.value + 1) % len(ColorMode))
            elif ch == ord("+"):
                time_delta /= 2
            elif ch == ord("-"):
                time_delta *= 2
            elif ch == ord("c"):
                botl.erase()
            elif ch == curses.KEY_MOUSE:
                _, x, y, *_ = curses.getmouse()
                x += pmincol
                y += pminrow
                area = funcs.square_at(bwgame.XY(x, y) * 32, 32)
                for u in funcs.find_units(area):
                    botl.addstr(
                        0,
                        10,
                        "%s: %s" % (u, unittypes.UNITTYPES[u.unit_type.id][0]),
                    )
                    break

    print("Thanks for playing.")


if __name__ == "__main__":
    main()
