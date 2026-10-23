"""Periodic-program optimizer (simulated annealing).

Idea: after the first pattern is accepted, the cheapest way to produce the next
ones is a short cycle of L emitter configurations that rebuilds every matter
cell (last event inside the cycle = "exactly 2 hits") and never leaves matter
on a skirting cell.  Then the program is
    setup (move emitters inactive to the cycle start, switch them on)
    + count * L steps.

Cells are Python-int bitsets, hit counts are computed with bit-sliced counters.
"""
import ctypes
import math
import os
import random
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, 'anneal.so')
_SRC = os.path.join(_HERE, 'anneal.c')
if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
    _tmp = _SO + '.%d.tmp' % os.getpid()
    subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', '-o', _tmp, _SRC, '-lm'])
    os.replace(_tmp, _SO)
_lib = ctypes.CDLL(_SO)
_lib.anneal.restype = ctypes.c_int
_lib.anneal.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.polish.restype = ctypes.c_int
_lib.polish.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
CW, CMAXL, CMAXP, CMAXT = 8, 32, 128, 1024
OPS_DELTA = {'-': 0, 'F': 1, 'B': -1}


def prog_to_states(cp, prog):
    plen = cp.plen
    pos, act = list(cp.home), [0] * NT
    traj = [[] for _ in range(NT)]
    for line in prog:
        for t in range(NT):
            if line[t] == 'T':
                act[t] = 1 - act[t]
            else:
                pos[t] = (pos[t] + OPS_DELTA[line[t]]) % plen
            traj[t].append((pos[t], act[t]))
    return traj


def states_to_prog(cp, traj, T):
    plen = cp.plen
    prog = []
    cur = [(cp.home[t], 0) for t in range(NT)]
    for i in range(T):
        line = []
        for t in range(NT):
            a, b = cur[t], traj[t][i]
            if a == b:
                line.append('-')
            elif a[1] != b[1]:
                line.append('T')
            elif (a[0] + 1) % plen == b[0]:
                line.append('F')
            else:
                line.append('B')
            cur[t] = b
        prog.append(line)
    return prog


def greedy_delete(cp, prog, simulate_fn, radius, grid):
    """Drop whole steps while the program stays legal and still produces `count` patterns."""
    k = 0
    while k < len(prog):
        traj = prog_to_states(cp, prog)
        prev = [(cp.home[t], 0) if k == 0 else traj[t][k - 1] for t in range(NT)]
        ok = k + 1 >= len(prog) or all(CycleProblem.adjacent(prev[t], traj[t][k + 1], cp.plen) for t in range(NT))
        if ok:
            new_traj = [traj[t][:k] + traj[t][k + 1:] for t in range(NT)]
            cand = states_to_prog(cp, new_traj, len(prog) - 1)
            if simulate_fn(radius, grid, cand) >= cp.count:
                prog = cand
                continue
        k += 1
    return prog


def polish(cp, prog, simulate_fn, radius, grid, iters=2000000, tries=3, seed=0):
    """Repeatedly try to find a valid program one step shorter (C annealer)."""
    _lib.set_count(cp.count)
    rng = random.Random(seed)
    best = greedy_delete(cp, prog, simulate_fn, radius, grid)
    fails = 0
    while fails < tries and len(best) > 1 and len(best) <= CMAXT:
        T = len(best) - 1
        traj = prog_to_states(cp, best)
        io_p = (ctypes.c_int * (NT * CMAXT))()
        io_a = (ctypes.c_int * (NT * CMAXT))()
        for t in range(NT):
            for i in range(T):
                io_p[t * CMAXT + i], io_a[t * CMAXT + i] = traj[t][i]
        c = _lib.polish(T, iters, rng.getrandbits(63), 30.0, 0.5, io_p, io_a)
        if c == 0:
            new = [[(io_p[t * CMAXT + i], io_a[t * CMAXT + i]) for i in range(T)] for t in range(NT)]
            cand = states_to_prog(cp, new, T)
            if simulate_fn(radius, grid, cand) >= cp.count:
                best = cand
                fails = 0
                continue
        fails += 1
    return best

NT = 6


def popcount(x):
    return bin(x).count('1')


class CycleProblem:
    def __init__(self, pb):
        self.pb = pb
        self.plen = pb.plen
        self.count = pb.count
        self.home = [int(p) for p in pb.start_pos]
        n = pb.n
        self.lines = []
        for t in range(NT):
            row = []
            for p in range(pb.plen):
                v = 0
                for i in range(n):
                    if pb.line[t, p, i]:
                        v |= 1 << i
                row.append(v)
            self.lines.append(row)
        self.S = sum(1 << int(i) for i in pb.S)
        self.K = sum(1 << int(i) for i in pb.K)
        # positions whose ray actually touches the interior (others are "parking" spots)
        self.useful = [[p for p in range(pb.plen) if self.lines[t][p]] for t in range(NT)]
        # partner[t][p] = position where track (t+3)%6 shoots along the same line
        self.partner = []
        for t in range(NT):
            u = (t + 3) % NT
            where = {self.lines[u][q]: q for q in self.useful[u]}
            self.partner.append({p: where.get(self.lines[t][p]) for p in self.useful[t]})

    def load_c(self):
        """Hand the problem to the C annealer."""
        def words(x):
            return [(x >> (64 * w)) & ((1 << 64) - 1) for w in range(CW)]
        lines = (ctypes.c_uint64 * (NT * CMAXP * CW))()
        for t in range(NT):
            for p in range(self.plen):
                for w, v in enumerate(words(self.lines[t][p])):
                    lines[(t * CMAXP + p) * CW + w] = v
        S = (ctypes.c_uint64 * CW)(*words(self.S))
        K = (ctypes.c_uint64 * CW)(*words(self.K))
        home = (ctypes.c_int * NT)(*self.home)
        partner = (ctypes.c_int * (NT * CMAXP))(*([-1] * (NT * CMAXP)))
        for t in range(NT):
            for p, q in self.partner[t].items():
                if q is not None:
                    partner[t * CMAXP + p] = q
        _lib.setup_problem(self.plen, lines, S, K, home, partner)

    def anneal_c(self, L, iters, seed, T0=60.0, T1=0.3):
        op = (ctypes.c_int * (NT * CMAXL))()
        oa = (ctypes.c_int * (NT * CMAXL))()
        best = _lib.anneal(L, iters, seed, T0, T1, op, oa)
        traj = [[(op[t * CMAXL + i], oa[t * CMAXL + i]) for i in range(L)] for t in range(NT)]
        return best, traj

    def circ(self, a, b):
        d = abs(a - b) % self.plen
        return min(d, self.plen - d)

    def steady_matter(self, traj, L):
        """traj[t] = list of (pos, act) of length L. Returns matter bitset after one cycle
        (cells without any event in the cycle are reported empty)."""
        m = 0
        for i in range(L):
            ones = twos = ge3 = 0
            for t in range(NT):
                p, a = traj[t][i]
                if not a:
                    continue
                x = self.lines[t][p]
                if not x:
                    continue
                ge3 |= twos & x
                twos = (twos & ~x) | (ones & x)
                ones = (ones & ~x) | (x & ~(ones | twos | ge3))
            m = (m & ~ge3) | twos
        return m

    def setup_cost(self, traj, L):
        return min(max(self.circ(self.home[t], traj[t][i][0]) + traj[t][i][1] for t in range(NT))
                   for i in range(L))

    def rotate_best(self, traj, L):
        """Rotate the cycle so that its last state is the cheapest one to reach."""
        costs = [max(self.circ(self.home[t], traj[t][i][0]) + traj[t][i][1] for t in range(NT))
                 for i in range(L)]
        k = (costs.index(min(costs)) + 1) % L
        return [tr[k:] + tr[:k] for tr in traj]

    def cost(self, traj, L):
        m = self.steady_matter(traj, L)
        viol = popcount(self.S & ~m) + popcount(self.K & m)
        return viol * 100 + self.setup_cost(traj, L), viol

    @staticmethod
    def adjacent(a, b, plen):
        if a == b:
            return True
        if a[1] == b[1]:
            d = (a[0] - b[0]) % plen
            return d == 1 or d == plen - 1
        return a[0] == b[0]

    def anneal(self, L, iters, rng, init=None):
        plen = self.plen
        if init is None:
            traj = []
            for t in range(NT):
                p = rng.choice(self.useful[t]) if self.useful[t] else self.home[t]
                traj.append([(p, 1)] * L)
        else:
            traj = [list(x) for x in init]
        cur, viol = self.cost(traj, L)
        best, best_traj = cur, [list(x) for x in traj]
        T0, T1 = 60.0, 0.3
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            t = rng.randrange(NT)
            old = traj[t]
            kind = rng.random()
            if kind < 0.15:  # shift the whole trajectory
                d = rng.choice((-1, 1))
                new = [((p + d) % plen, a) for p, a in old]
            elif kind < 0.22:  # jump: constant state somewhere useful
                p = rng.choice(self.useful[t]) if self.useful[t] else old[0][0]
                new = [(p, rng.random() < 0.8)] * L
            elif kind < 0.27:  # flip activity everywhere
                new = [(p, 1 - a) for p, a in old]
            elif kind < 0.33:  # make the opposite track mirror this one (double line)
                u = (t + 3) % NT
                pm = self.partner[t]
                if any(pm.get(p) is None for p, a in old):
                    continue
                old = traj[u]
                t = u
                new = [(pm[p], a) for p, a in traj[(u + 3) % NT]]
            else:  # change one step
                i = rng.randrange(L)
                prv, nxt = old[i - 1], old[(i + 1) % L]
                p0, a0 = prv
                cands = [(p0, a0), ((p0 + 1) % plen, a0), ((p0 - 1) % plen, a0), (p0, 1 - a0)]
                cands = [c for c in cands if c != old[i] and self.adjacent(c, nxt, plen)]
                if L == 1:
                    p, a = old[0]
                    cands = [((p + 1) % plen, a), ((p - 1) % plen, a), (p, 1 - a)]
                if not cands:
                    continue
                c = rng.choice(cands)
                new = list(old)
                new[i] = c
            traj[t] = new
            c, v = self.cost(traj, L)
            if c <= cur or rng.random() < math.exp((cur - c) / T):
                cur = c
                if c < best:
                    best, best_traj = c, [list(x) for x in traj]
            else:
                traj[t] = old
        return best, best_traj

    def program(self, traj, L, cycles):
        """Build the instruction lines: setup + `cycles` repetitions of the cycle."""
        plen = self.plen
        prog = []
        traj = self.rotate_best(traj, L)
        last = [traj[t][L - 1] for t in range(NT)]
        setup = self.setup_cost(traj, L)
        pos = list(self.home)
        for step in range(setup):
            line = []
            for t in range(NT):
                tp, ta = last[t]
                if ta and step == setup - 1:
                    line.append('T')
                    continue
                if pos[t] == tp:
                    line.append('-')
                    continue
                fwd = (tp - pos[t]) % plen
                if fwd <= plen - fwd:
                    line.append('F')
                    pos[t] = (pos[t] + 1) % plen
                else:
                    line.append('B')
                    pos[t] = (pos[t] - 1) % plen
            prog.append(line)
        cur = list(last)
        for _ in range(cycles):
            for i in range(L):
                line = []
                for t in range(NT):
                    a, b = cur[t], traj[t][i]
                    if a == b:
                        line.append('-')
                    elif a[1] != b[1]:
                        line.append('T')
                    elif (a[0] + 1) % plen == b[0]:
                        line.append('F')
                    else:
                        line.append('B')
                    cur[t] = b
                prog.append(line)
        return prog


def solve_cycle(pb, simulate_fn, radius, grid, max_L=CMAXL, iters=20000, restarts=2, seed=0, budget_best=None):
    """Returns the shortest valid program found by cycle annealing, or None."""
    cp = CycleProblem(pb)
    cp.load_c()
    rng = random.Random(seed)
    best_prog = None
    best_len = budget_best if budget_best is not None else 10 ** 9
    for L in range(1, max_L + 1):
        if 1 + pb.count * L >= best_len:
            break
        found = None
        for r in range(restarts):
            c, traj = cp.anneal_c(L, iters, rng.getrandbits(63))
            if c < 100 and (found is None or c < found[0]):
                found = (c, traj)
        if found is None:
            continue
        traj = found[1]
        prog = cp.program(traj, L, pb.count + 2)
        # trim to the step where the required count is reached
        acc = simulate_prefix(simulate_fn, radius, grid, prog, pb.count)
        if acc is not None and acc < best_len:
            best_len, best_prog = acc, prog[:acc]
    return best_prog


def simulate_prefix(simulate_fn, radius, grid, prog, count):
    """Smallest prefix length whose simulation accepts `count` patterns (binary search)."""
    if simulate_fn(radius, grid, prog) < count:
        return None
    lo, hi = 1, len(prog)
    while lo < hi:
        mid = (lo + hi) // 2
        if simulate_fn(radius, grid, prog[:mid]) >= count:
            hi = mid
        else:
            lo = mid + 1
    return lo
