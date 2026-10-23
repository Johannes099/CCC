"""Geometry, I/O and an exact simulator for the CCC Nanofabricator (level 6).

The simulator is a direct port of the JavaScript in sandbox-level-6.html.
"""

# (dc, dr) in overlay coordinates, same order as FIRE_OV in the sandbox
FIRE = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]  # E SE SW W NW NE
DIR_TO = FIRE
PERIM_DIRS = [(1, -1), (2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1)]  # NE E SE SW W NW


class Geometry:
    def __init__(self, board_radius):
        R = board_radius + 1
        self.R = R
        self.rows = 2 * R - 1
        self.cols = 4 * R - 3
        cols = self.cols

        perimeter = []
        if R > 1:
            c, r = 0, R - 1
            for dc, dr in PERIM_DIRS:
                for _ in range(R - 1):
                    perimeter.append((c, r))
                    c, r = c + dc, r + dr
        self.perimeter = perimeter
        self.plen = len(perimeter)

        def in_grid(c, r):
            if r < 0 or r >= 2 * R - 1:
                return False
            if (r + c + R + 1) % 2 != 0:
                return False
            if r < R:
                return c + r >= R - 1 and c - r <= cols - R + 1
            return c + (2 * R - 1 - r) >= R - 1 and c + r - 2 * R + 1 < cols - R + 1

        base = []
        for r in range(self.rows):
            r_rail = r == 0 or r == self.rows - 1
            row = []
            for c in range(cols):
                if (r + c + R + 1) % 2 != 0:
                    row.append('-')
                    continue
                if r < R:
                    c_rail = c == R - 1 - r or c == cols - 1 - R + 1 + r
                    inside = c + r >= R - 1 and c - r <= cols - R + 1
                else:
                    c_rail = c == R - 1 + r - 2 * R + 2 or c == cols - 1 + R - 1 - r
                    inside = c + 2 * R - 1 - r >= R - 1 and c + r - 2 * R + 1 < cols - R + 1
                row.append(('#' if (r_rail or c_rail) else 'O') if inside else ':')
            base.append(row)
        self.base = base

        # ray cells per (perimeter index, track)
        self.rays = []
        for (c0, r0) in perimeter:
            per_track = []
            for dc, dr in FIRE:
                cells = []
                c, r = c0 + dc, r0 + dr
                while in_grid(c, r):
                    cells.append((r, c))
                    c, r = c + dc, r + dr
                per_track.append(cells)
            self.rays.append(per_track)

        # interior cells (the only ones that can hold matter)
        self.cells = [(r, c) for r in range(self.rows) for c in range(cols) if base[r][c] == 'O']
        self.cell_index = {rc: i for i, rc in enumerate(self.cells)}


def read_input(path):
    lines = open(path).read().split('\n')
    n = int(lines[0])
    i = 1
    cases = []
    for _ in range(n):
        while not lines[i].strip():
            i += 1
        radius, rows, cols, count = map(int, lines[i].split())
        grid = [list(lines[i + 1 + k]) for k in range(rows)]
        i += 1 + rows
        cases.append((radius, rows, cols, count, grid))
    return cases


def write_output(path, programs):
    with open(path, 'w') as f:
        f.write('\n\n'.join('\n'.join(' '.join(line) for line in prog) for prog in programs))
        f.write('\n')


def simulate(radius, pattern, program):
    """Returns the number of accepted patterns (exact port of the sandbox)."""
    g = Geometry(radius)
    plen = g.plen
    grid = [row[:] for row in g.base]
    indices = [t * max(g.R - 1, 1) for t in range(6)]
    active = [False] * 6
    rows, cols = len(pattern), len(pattern[0])

    def is_skirting(r, c):
        for dc, dr in DIR_TO:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and pattern[rr][cc] == 'M':
                return True
        return False

    m_cells = [(r, c) for r in range(rows) for c in range(cols) if pattern[r][c] == 'M']
    k_cells = [(r, c) for r in range(rows) for c in range(cols)
               if pattern[r][c] == 'O' and is_skirting(r, c)]
    accepted = 0
    for step in program:
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] in ('x', 'd'):
                    grid[r][c] = 'O'
                elif grid[r][c] == 'm':
                    grid[r][c] = 'M'
        for t in range(6):
            if step[t] == 'F':
                indices[t] = (indices[t] + 1) % plen
            elif step[t] == 'B':
                indices[t] = (indices[t] + plen - 1) % plen
            elif step[t] == 'T':
                active[t] = not active[t]
        hits = {}
        for t in range(6):
            if not active[t]:
                continue
            for rc in g.rays[indices[t]][t]:
                hits[rc] = hits.get(rc, 0) + 1
        for (r, c), n in hits.items():
            cell = grid[r][c]
            if cell not in ('O', 'M'):
                continue
            if n == 2:
                grid[r][c] = 'm' if cell == 'O' else 'M'
            elif n >= 3:
                grid[r][c] = 'd'
        if len(grid) == rows and len(grid[0]) == cols:
            if all(grid[r][c] in ('M', 'm') for r, c in m_cells) and \
               not any(grid[r][c] in ('M', 'm') for r, c in k_cells):
                for r, c in m_cells:
                    grid[r][c] = 'x'
                accepted += 1
    return accepted
