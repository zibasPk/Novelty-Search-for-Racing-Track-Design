"""
Python version of Alea PRNG
"""


import time

class s:
    def __init__(self, t):
        if t is None:
            t = time.time() * 1000  # Equivalent to +new Date

        s = 4022871197

        def e(t):
            nonlocal s
            t = str(t)
            for k in range(len(t)):
                s += ord(t[k])
                i = 0.02519603282416938 * s
                s = int(i) & 0xFFFFFFFF
                i -= s
                i *= s
                s = int(i) & 0xFFFFFFFF
                i -= s
                s += 4294967296 * i
            return 2.3283064365386963e-10 * (int(s) & 0xFFFFFFFF)

        self.c = 1
        self.s0 = e(" ")
        self.s1 = e(" ")
        self.s2 = e(" ")
        
        self.s0 -= e(t)
        if self.s0 < 0: self.s0 += 1
        
        self.s1 -= e(t)
        if self.s1 < 0: self.s1 += 1
        
        self.s2 -= e(t)
        if self.s2 < 0: self.s2 += 1

    def next(self):
        t = self.c
        s = self.s0
        e = self.s1
        i = self.s2
        
        h = 2091639 * s + 2.3283064365386963e-10 * t
        
        self.s0 = e
        self.s1 = i
        self.c = int(h)
        self.s2 = h - self.c
        
        return self.s2

    def copy(self, t, s):
        s.c = t.c
        s.s0 = t.s0
        s.s1 = t.s1
        s.s2 = t.s2
        return s

class _Obj:
    pass

def t(t, e=None):
    i = s(t)
    
    def h():
        return i.next()

    def _double():
        return h() + 11102230246251565e-32 * int(2097152 * h())
    h.double = _double

    def _int32():
        val = int(4294967296 * i.next()) & 0xFFFFFFFF
        return val - 0x100000000 if val > 0x7FFFFFFF else val
    h.int32 = _int32

    h.quick = h

    def _anon(t, s, e):
        i = getattr(e, 'state', None) if e else None
        if i:
            s.copy(i, s)
            def _state():
                return s.copy(s, _Obj())
            t.state = _state
            
    _anon(h, i, e)

    return h


prng_alea = t

__all__ = ["prng_alea"]