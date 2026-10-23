// Simulated annealing core for cycle.py (compiled automatically to anneal.so).
// Cells are bitsets of W 64-bit words.
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NT 6
#define W 8
#define MAXL 32
#define MAXP 128

static int plen, L;
static uint64_t lines[NT][MAXP][W];
static uint64_t S[W], K[W];
static int home[NT];
static int partner[NT][MAXP];
static int useful[NT][MAXP], nuseful[NT];

static uint64_t rs;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double rndf(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

static inline int circ(int a, int b) { int d = a - b; if (d < 0) d = -d; d %= plen; return d < plen - d ? d : plen - d; }

static int cost(int pos[NT][MAXL], int act[NT][MAXL], int *violp) {
    uint64_t m[W] = {0};
    for (int i = 0; i < L; i++) {
        uint64_t ones[W] = {0}, twos[W] = {0}, ge3[W] = {0};
        for (int t = 0; t < NT; t++) {
            if (!act[t][i]) continue;
            uint64_t *x = lines[t][pos[t][i]];
            for (int w = 0; w < W; w++) {
                uint64_t xv = x[w];
                ge3[w] |= twos[w] & xv;
                uint64_t nt = (twos[w] & ~xv) | (ones[w] & xv);
                ones[w] = (ones[w] & ~xv) | (xv & ~(ones[w] | twos[w] | ge3[w]));
                twos[w] = nt;
            }
        }
        for (int w = 0; w < W; w++) m[w] = (m[w] & ~ge3[w]) | twos[w];
    }
    int viol = 0;
    for (int w = 0; w < W; w++) viol += __builtin_popcountll(S[w] & ~m[w]) + __builtin_popcountll(K[w] & m[w]);
    int setup = 1 << 20;  // best rotation of the cycle
    for (int i = 0; i < L; i++) {
        int s = 0;
        for (int t = 0; t < NT; t++) {
            int c = circ(home[t], pos[t][i]) + act[t][i];
            if (c > s) s = c;
        }
        if (s < setup) setup = s;
    }
    *violp = viol;
    return viol * 100 + setup;
}

static inline int adjacent(int p1, int a1, int p2, int a2) {
    if (p1 == p2 && a1 == a2) return 1;
    if (a1 == a2) { int d = ((p1 - p2) % plen + plen) % plen; return d == 1 || d == plen - 1; }
    return p1 == p2;
}

void setup_problem(int plen_, const uint64_t *lines_, const uint64_t *S_, const uint64_t *K_,
                   const int *home_, const int *partner_) {
    plen = plen_;
    memcpy(lines, lines_, sizeof(lines));
    memcpy(S, S_, sizeof(S));
    memcpy(K, K_, sizeof(K));
    memcpy(home, home_, sizeof(home));
    memcpy(partner, partner_, sizeof(partner));
    for (int t = 0; t < NT; t++) {
        nuseful[t] = 0;
        for (int p = 0; p < plen; p++) {
            int any = 0;
            for (int w = 0; w < W; w++) any |= lines[t][p][w] != 0;
            if (any) useful[t][nuseful[t]++] = p;
        }
    }
}

// Returns best cost; best trajectory written to out_pos/out_act ([NT][MAXL]).
int anneal(int L_, long iters, uint64_t seed, double T0, double T1, int *out_pos, int *out_act) {
    L = L_;
    rs = seed * 2654435761ULL + 88172645463325252ULL;
    static int pos[NT][MAXL], act[NT][MAXL], bpos[NT][MAXL], bact[NT][MAXL];
    int npos[MAXL], nact[MAXL], opos[MAXL], oact[MAXL];
    for (int t = 0; t < NT; t++) {
        int p = nuseful[t] ? useful[t][rnd() % nuseful[t]] : home[t];
        for (int i = 0; i < L; i++) { pos[t][i] = p; act[t][i] = 1; }
    }
    int viol;
    int cur = cost(pos, act, &viol), best = cur;
    memcpy(bpos, pos, sizeof(pos)); memcpy(bact, act, sizeof(act));
    double lr = log(T1 / T0);
    for (long it = 0; it < iters; it++) {
        double T = T0 * exp(lr * (double)it / (double)iters);
        int t = rnd() % NT;
        double k = rndf();
        int tt = t;
        if (k < 0.15) {
            int d = (rnd() & 1) ? 1 : plen - 1;
            for (int i = 0; i < L; i++) { npos[i] = (pos[t][i] + d) % plen; nact[i] = act[t][i]; }
        } else if (k < 0.22) {
            int p = nuseful[t] ? useful[t][rnd() % nuseful[t]] : pos[t][0];
            int a = rndf() < 0.8;
            for (int i = 0; i < L; i++) { npos[i] = p; nact[i] = a; }
        } else if (k < 0.27) {
            for (int i = 0; i < L; i++) { npos[i] = pos[t][i]; nact[i] = 1 - act[t][i]; }
        } else if (k < 0.33) {
            int u = (t + 3) % NT, ok = 1;
            for (int i = 0; i < L; i++) {
                int q = partner[t][pos[t][i]];
                if (q < 0) { ok = 0; break; }
                npos[i] = q; nact[i] = act[t][i];
            }
            if (!ok) continue;
            tt = u;
        } else if (k < 0.45 && L > 2) {
            // time shift: wait at i, drop step j (cyclic); or shift positions of a segment
            int i = rnd() % L, j = rnd() % L;
            for (int q = 0; q < L; q++) { npos[q] = pos[t][q]; nact[q] = act[t][q]; }
            if (rndf() < 0.6) {
                if (i == j) continue;
                int q = j;
                while (q != i) { int pq = (q + L - 1) % L; npos[q] = npos[pq]; nact[q] = nact[pq]; q = pq; }
            } else {
                int d = (rnd() & 1) ? 1 : plen - 1;
                int q = i;
                while (1) { npos[q] = (npos[q] + d) % plen; if (q == j) break; q = (q + 1) % L; }
            }
            int ok = 1;
            for (int q = 0; q < L && ok; q++) ok = adjacent(npos[q], nact[q], npos[(q + 1) % L], nact[(q + 1) % L]);
            if (!ok) continue;
        } else {
            int i = rnd() % L;
            for (int j = 0; j < L; j++) { npos[j] = pos[t][j]; nact[j] = act[t][j]; }
            if (L == 1) {
                int r = rnd() % 3;
                if (r == 0) npos[0] = (pos[t][0] + 1) % plen;
                else if (r == 1) npos[0] = (pos[t][0] + plen - 1) % plen;
                else nact[0] = 1 - act[t][0];
            } else {
                int ip = (i + L - 1) % L, in = (i + 1) % L;
                int p0 = pos[t][ip], a0 = act[t][ip];
                int cp[4] = {p0, (p0 + 1) % plen, (p0 + plen - 1) % plen, p0};
                int ca[4] = {a0, a0, a0, 1 - a0};
                int cand[4], nc = 0;
                for (int c = 0; c < 4; c++) {
                    if (cp[c] == pos[t][i] && ca[c] == act[t][i]) continue;
                    if (!adjacent(cp[c], ca[c], pos[t][in], act[t][in])) continue;
                    cand[nc++] = c;
                }
                if (!nc) continue;
                int c = cand[rnd() % nc];
                npos[i] = cp[c]; nact[i] = ca[c];
            }
        }
        memcpy(opos, pos[tt], sizeof(int) * L); memcpy(oact, act[tt], sizeof(int) * L);
        memcpy(pos[tt], npos, sizeof(int) * L); memcpy(act[tt], nact, sizeof(int) * L);
        int c = cost(pos, act, &viol);
        if (c <= cur || rndf() < exp((cur - c) / T)) {
            cur = c;
            if (c < best) { best = c; memcpy(bpos, pos, sizeof(pos)); memcpy(bact, act, sizeof(act)); }
        } else {
            memcpy(pos[tt], opos, sizeof(int) * L); memcpy(act[tt], oact, sizeof(int) * L);
        }
    }
    for (int t = 0; t < NT; t++)
        for (int i = 0; i < MAXL; i++) { out_pos[t * MAXL + i] = bpos[t][i]; out_act[t * MAXL + i] = bact[t][i]; }
    return best;
}

// ---------------------------------------------------------------------------
// Polishing a whole (non-cyclic) program of fixed length T: feasible iff the real
// simulation accepts `count` patterns within T steps.
#define MAXT 1024
static int count_req;
static int TT;
static int ppos[NT][MAXT], pact[NT][MAXT];

void set_count(int c) { count_req = c; }

static int prog_cost(void) {
    uint64_t m[W] = {0};
    int acc = 0;
    for (int i = 0; i < TT; i++) {
        uint64_t ones[W] = {0}, twos[W] = {0}, ge3[W] = {0};
        for (int t = 0; t < NT; t++) {
            if (!pact[t][i]) continue;
            uint64_t *x = lines[t][ppos[t][i]];
            for (int w = 0; w < W; w++) {
                uint64_t xv = x[w];
                ge3[w] |= twos[w] & xv;
                uint64_t nt = (twos[w] & ~xv) | (ones[w] & xv);
                ones[w] = (ones[w] & ~xv) | (xv & ~(ones[w] | twos[w] | ge3[w]));
                twos[w] = nt;
            }
        }
        int ok = 1;
        for (int w = 0; w < W; w++) {
            m[w] = (m[w] & ~ge3[w]) | twos[w];
            if ((S[w] & ~m[w]) || (K[w] & m[w])) ok = 0;
        }
        if (ok) {
            acc++;
            if (acc >= count_req) return 0;
            for (int w = 0; w < W; w++) m[w] &= ~S[w];
        }
    }
    int viol = 0;
    for (int w = 0; w < W; w++) viol += __builtin_popcountll(S[w] & ~m[w]) + __builtin_popcountll(K[w] & m[w]);
    return (count_req - acc) * 1000 + viol * 10;
}

static int track_valid(int t) {
    int p = home[t], a = 0;
    for (int i = 0; i < TT; i++) {
        if (!adjacent(p, a, ppos[t][i], pact[t][i])) return 0;
        p = ppos[t][i]; a = pact[t][i];
    }
    return 1;
}

// in/out arrays are [NT][MAXT]
int polish(int T, long iters, uint64_t seed, double T0, double T1, int *io_pos, int *io_act) {
    TT = T;
    rs = seed * 2654435761ULL + 88172645463325252ULL;
    for (int t = 0; t < NT; t++)
        for (int i = 0; i < T; i++) { ppos[t][i] = io_pos[t * MAXT + i]; pact[t][i] = io_act[t * MAXT + i]; }
    int opos[MAXT], oact[MAXT];
    int cur = prog_cost(), best = cur;
    if (best == 0) return 0;
    double lr = log(T1 / T0);
    for (long it = 0; it < iters; it++) {
        double Tm = T0 * exp(lr * (double)it / (double)iters);
        double k = rndf();
        if (k < 0.08) {  // global: delete step i (append a wait) or insert a wait at i (drop last)
            static int sp[NT][MAXT], sa[NT][MAXT];
            memcpy(sp, ppos, sizeof(sp)); memcpy(sa, pact, sizeof(sa));
            int i = rnd() % T;
            if (rnd() & 1) {
                for (int tt = 0; tt < NT; tt++)
                    for (int q = i; q < T - 1; q++) { ppos[tt][q] = ppos[tt][q + 1]; pact[tt][q] = pact[tt][q + 1]; }
            } else {
                for (int tt = 0; tt < NT; tt++)
                    for (int q = T - 1; q > i; q--) { ppos[tt][q] = ppos[tt][q - 1]; pact[tt][q] = pact[tt][q - 1]; }
            }
            int ok = 1;
            for (int tt = 0; tt < NT && ok; tt++) ok = track_valid(tt);
            int c = ok ? prog_cost() : 1 << 30;
            if (ok && (c <= cur || rndf() < exp((cur - c) / Tm))) {
                cur = c;
                if (c < best) {
                    best = c;
                    for (int tt = 0; tt < NT; tt++)
                        for (int q = 0; q < T; q++) { io_pos[tt * MAXT + q] = ppos[tt][q]; io_act[tt * MAXT + q] = pact[tt][q]; }
                    if (best == 0) return 0;
                }
            } else {
                memcpy(ppos, sp, sizeof(sp)); memcpy(pact, sa, sizeof(sa));
            }
            continue;
        }
        int t = rnd() % NT;
        memcpy(opos, ppos[t], sizeof(int) * T); memcpy(oact, pact[t], sizeof(int) * T);
        if (k < 0.5) {  // change one step
            int i = rnd() % T;
            int r = rnd() % 4;
            int p = ppos[t][i];
            if (r == 0) ppos[t][i] = (p + 1) % plen;
            else if (r == 1) ppos[t][i] = (p + plen - 1) % plen;
            else if (r == 2) pact[t][i] = 1 - pact[t][i];
            else { int j = i ? i - 1 : -1; if (j < 0) { ppos[t][i] = home[t]; pact[t][i] = 0; } else { ppos[t][i] = ppos[t][j]; pact[t][i] = pact[t][j]; } }
        } else if (k < 0.75) {  // insert a wait at i, drop step j > i (or the reverse)
            int i = rnd() % T, j = rnd() % T;
            if (i == j) continue;
            if (i < j) {
                for (int q = j; q > i; q--) { ppos[t][q] = ppos[t][q - 1]; pact[t][q] = pact[t][q - 1]; }
            } else {
                for (int q = j; q < i; q++) { ppos[t][q] = ppos[t][q + 1]; pact[t][q] = pact[t][q + 1]; }
            }
        } else if (k < 0.9) {  // shift a suffix by +-1
            int i = rnd() % T, d = (rnd() & 1) ? 1 : plen - 1;
            for (int q = i; q < T; q++) ppos[t][q] = (ppos[t][q] + d) % plen;
        } else {  // flip activity on a suffix
            int i = rnd() % T;
            for (int q = i; q < T; q++) pact[t][q] = 1 - pact[t][q];
        }
        if (!track_valid(t)) { memcpy(ppos[t], opos, sizeof(int) * T); memcpy(pact[t], oact, sizeof(int) * T); continue; }
        int c = prog_cost();
        if (c <= cur || rndf() < exp((cur - c) / Tm)) {
            cur = c;
            if (c < best) {
                best = c;
                for (int tt = 0; tt < NT; tt++)
                    for (int q = 0; q < T; q++) { io_pos[tt * MAXT + q] = ppos[tt][q]; io_act[tt * MAXT + q] = pact[tt][q]; }
                if (best == 0) return 0;
            }
        } else {
            memcpy(ppos[t], opos, sizeof(int) * T); memcpy(pact[t], oact, sizeof(int) * T);
        }
    }
    return best;
}
