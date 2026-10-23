"""CCC level 6 optimizer: beam search over emitter instructions.

Every step each of the 6 emitters picks one of {-, F, B, T}, giving 4^6 = 4096
successor configurations.  All of them are evaluated at once with numpy:
  * the hit count of every interior cell (2 hits -> matter, >=3 -> destroyed),
  * pattern acceptance,
  * a heuristic: for every missing matter cell the cost of creating it is the
    second-smallest "distance" of any emitter to a ray through that cell
    (two rays are needed), plus a penalty for matter on skirting cells.
The beam keeps the best distinct states; the search stops at the first step
where the required number of patterns has been accepted.

Usage: python3 solver.py [level files...] [--mode fast|good] [--jobs J]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from hexfab import Geometry, DIR_TO, read_input, write_output, simulate

OPS = ['-', 'F', 'B', 'T']
NT = 6


class Problem:
    def __init__(self, radius, grid, count):
        g = Geometry(radius)
        self.g = g
        self.count = count
        n = len(g.cells)
        self.n = n
        plen = g.plen
        self.plen = plen
        rows, cols = len(grid), len(grid[0])

        is_m = np.zeros(n, bool)
        is_k = np.zeros(n, bool)
        for i, (r, c) in enumerate(g.cells):
            if grid[r][c] == 'M':
                is_m[i] = True
            elif grid[r][c] == 'O':
                for dc, dr in DIR_TO:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == 'M':
                        is_k[i] = True
        self.S = np.nonzero(is_m)[0]
        self.K = np.nonzero(is_k)[0]

        # line[t, p] = interior cells hit by track t standing at perimeter index p
        line = np.zeros((NT, plen, n), np.uint8)
        target = np.full((NT, n), -1, int)  # perimeter index whose ray covers cell
        for p in range(plen):
            for t in range(NT):
                for rc in g.rays[p][t]:
                    i = g.cell_index.get(rc)
                    if i is not None:
                        line[t, p, i] = 1
                        target[t, i] = p
        self.line = line
        self.target = target
        idx = np.arange(plen)
        diff = np.abs(idx[:, None] - idx[None, :])
        circ = np.minimum(diff, plen - diff)
        # dist[t, p, s] = moves for track t at p to reach a ray through matter cell s
        # (matter cells first, then skirting cells)
        self.SK = np.concatenate([self.S, self.K])
        self.dist = np.stack([circ[:, target[t, self.SK]] for t in range(NT)]).astype(np.int16)

        # option o of track t: new position / active flag
        self.start_pos = np.array([t * max(g.R - 1, 1) for t in range(NT)])


def expand_tracks(pb, pos, act):
    """Per-track arrays over the 4 options: hits [4, n] and distance [4, |S|+|K|]."""
    plen = pb.plen
    hits, dists, npos, nact = [], [], [], []
    for t in range(NT):
        p, a = pos[t], act[t]
        ps = [p, (p + 1) % plen, (p - 1) % plen, p]
        as_ = [a, a, a, 1 - a]
        h = np.stack([pb.line[t, ps[o]] * as_[o] for o in range(4)])
        d = np.stack([pb.dist[t, ps[o]] + (1 - as_[o]) for o in range(4)])
        hits.append(h)
        dists.append(d)
        npos.append(ps)
        nact.append(as_)
    return hits, dists, npos, nact


def bshape(t, extra):
    s = [1] * NT
    s[t] = 4
    return s + [extra]


def evaluate(pb, pos, act, matter, accepted, wK, rng, noise, wN=0.0):
    hits, dists, npos, nact = expand_tracks(pb, pos, act)
    n, nS = pb.n, len(pb.S)
    H = hits[0].reshape(bshape(0, n))
    for t in range(1, NT):
        H = H + hits[t].reshape(bshape(t, n))
    H = H.reshape(4 ** NT, n)
    new_m = (matter[None, :] & (H < 3)) | (H == 2)
    mS = new_m[:, pb.S]
    acc = mS.all(1)
    if len(pb.K):
        mK = new_m[:, pb.K]
        acc &= ~mK.any(1)
    if acc.any():
        new_m[np.ix_(acc, pb.S)] = False
        mS = new_m[:, pb.S]

    # 2nd / 3rd smallest distance over tracks for every matter / skirting cell
    nSK = len(pb.SK)
    m1 = dists[0].reshape(bshape(0, nSK)).astype(np.int16)
    m2 = np.full_like(m1, 999)
    m3 = np.full_like(m1, 999)
    for t in range(1, NT):
        d = dists[t].reshape(bshape(t, nSK))
        m3 = np.minimum(m3, np.maximum(m2, d))
        m2 = np.minimum(m2, np.maximum(m1, d))
        m1 = np.minimum(m1, d)
    m2 = np.broadcast_to(m2[..., :nS], [4] * NT + [nS]).reshape(4 ** NT, nS)
    missing = np.where(mS, 999, m2)
    cost = np.where(mS, 0, 1 + m2).sum(1)
    if wN:
        # pull towards the nearest missing cell so the beam never stalls on a plateau
        near = missing.min(1)
        cost = cost + wN * np.where(near == 999, 0, near)
    if len(pb.K):
        m3 = np.broadcast_to(m3[..., nS:], [4] * NT + [len(pb.K)]).reshape(4 ** NT, len(pb.K))
        cost = cost + np.where(mK, wK + 2 * m3, 0).sum(1)

    acc_total = accepted + acc
    score = acc_total * 100000.0 - cost
    if noise:
        score = score + rng.random(score.shape) * noise
    return score, new_m, acc_total, npos, nact


def decode(combo):
    ops = []
    for t in range(NT - 1, -1, -1):
        ops.append(combo % 4)
        combo //= 4
    return ops[::-1]


def solve(pb, beam=16, wK=8.0, noise=0.5, seed=0, max_steps=3000, stall=150, wN=0.0):
    rng = np.random.default_rng(seed)
    pos0 = tuple(int(x) for x in pb.start_pos)
    states = [(pos0, (0,) * NT, np.zeros(pb.n, bool), 0, None)]  # pos, act, matter, accepted, hist
    last_acc, last_step = 0, 0
    for step in range(max_steps):
        top_acc = max(s[3] for s in states)
        if top_acc > last_acc:
            last_acc, last_step = top_acc, step
        elif step - last_step > stall:
            break
        cand = []
        for si, (pos, act, matter, accepted, hist) in enumerate(states):
            score, new_m, acc_total, npos, nact = evaluate(pb, pos, act, matter, accepted, wK, rng, noise, wN)
            done = np.nonzero(acc_total >= pb.count)[0]
            if len(done):
                hist = (decode(int(done[0])), hist)
                prog = []
                while hist is not None:
                    prog.append([OPS[o] for o in hist[0]])
                    hist = hist[1]
                return prog[::-1]
            k = min(len(score), beam * 4)
            top = np.argpartition(-score, k - 1)[:k]
            for c in top:
                cand.append((score[c], si, int(c), new_m[c], int(acc_total[c]), npos, nact))
        cand.sort(key=lambda x: -x[0])
        seen = set()
        nxt = []
        for sc, si, c, m, acc, npos, nact in cand:
            ops = decode(c)
            p = tuple(npos[t][ops[t]] for t in range(NT))
            a = tuple(nact[t][ops[t]] for t in range(NT))
            key = (p, a, m.tobytes(), acc)
            if key in seen:
                continue
            seen.add(key)
            nxt.append((p, a, m, acc, (ops, states[si][4])))
            if len(nxt) >= beam:
                break
        states = nxt
    raise RuntimeError('no solution found')


def fallback(pb):
    """Always-valid plan: only tracks 0 (E) and 1 (SE) are used, so at most one cell
    (their crossing) gets 2 hits per step; cells are built one by one."""
    plen = pb.plen
    pos = [int(p) for p in pb.start_pos]
    act = [0] * NT
    prog = []

    def toward(p, q):
        if p == q:
            return '-', p
        fwd = (q - p) % plen
        return ('F', (p + 1) % plen) if fwd <= plen - fwd else ('B', (p - 1) % plen)

    prog.append(['-', 'T', '-', '-', '-', '-'])  # track 1 is always on, alone it creates nothing
    act[1] = 1
    for _ in range(pb.count):
        todo = [int(s) for s in pb.S]
        while todo:
            def d(s):
                a, b = pb.target[0, s], pb.target[1, s]
                da = min((a - pos[0]) % plen, (pos[0] - a) % plen)
                db = min((b - pos[1]) % plen, (pos[1] - b) % plen)
                return max(da, db)
            s = min(todo, key=d)
            todo.remove(s)
            a, b = int(pb.target[0, s]), int(pb.target[1, s])
            while True:
                line = ['-'] * NT
                if act[0]:
                    line[0] = 'T'
                    act[0] = 0
                    line[1], pos[1] = toward(pos[1], b)
                elif pos[0] != a or pos[1] != b:
                    line[0], pos[0] = toward(pos[0], a)
                    line[1], pos[1] = toward(pos[1], b)
                else:
                    line[0] = 'T'
                    act[0] = 1
                    prog.append(line)
                    break
                prog.append(line)
    return prog


# (beam, wK, noise) settings tried per case; the shortest program wins
SETTINGS = {
    'fast': [(16, 8.0, 0.0), (16, 8.0, 1.0), (16, 4.0, 2.0), (32, 16.0, 1.0)],
    'good': [(16, 8.0, 0.0), (32, 8.0, 0.5), (32, 4.0, 1.0), (64, 8.0, 0.5), (64, 16.0, 1.0)],
}


def solve_case(args):
    radius, grid, count, mode = args
    pb = Problem(radius, grid, count)
    best = None
    for seed, (beam, wK, noise) in enumerate(SETTINGS[mode]):
        if mode == 'fast' and best is not None:
            break
        try:
            prog = solve(pb, beam=beam, wK=wK, noise=noise, seed=seed)
        except RuntimeError:
            continue
        assert simulate(radius, grid, prog) >= count
        if best is None or len(prog) < len(best):
            best = prog
    if best is None:
        best = fallback(pb)
        assert simulate(radius, grid, best) >= count
    return best


def main():
    args = sys.argv[1:]
    mode, jobs = 'good', os.cpu_count()
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--mode':
            mode = args[i + 1]
            i += 2
        elif args[i] == '--jobs':
            jobs = int(args[i + 1])
            i += 2
        else:
            files.append(args[i])
            i += 1
    here = os.path.dirname(os.path.abspath(__file__))
    for f in files:
        t0 = time.time()
        cases = read_input(f)
        work = [(rad, grid, cnt, mode) for rad, rows, cols, cnt, grid in cases]
        with Pool(jobs) as pool:
            progs = pool.map(solve_case, work, chunksize=1)
        out = os.path.join(here, 'output', os.path.basename(f).replace('in_', 'out_'))
        write_output(out, progs)
        score = sum(len(p) for p in progs)
        print(f'{os.path.basename(f)}: score {score} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
