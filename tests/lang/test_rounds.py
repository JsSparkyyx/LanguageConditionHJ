"""RoundSchedule: global (task, embodiment) cursor, T1 host cursor per embodiment, state round trip."""

from collections import Counter

from main.lang.config import CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS
from main.lang.rounds import RoundSchedule


def test_schedule_balanced_and_resumable():
    s = RoundSchedule(CANON_TASKS, CANON_EMBODIMENTS, CANON_T1_HOSTS)
    slots = [s.next() for _ in range(14 * 5)]
    assert set(Counter((t, e) for t, e, _ in slots).values()) == {5}
    hosts = Counter((e, h) for t, e, h in slots if t == "selective_avoid")
    assert set(hosts.values()) == {1} and len(hosts) == 10   # every host once per embodiment
    st = s.state()
    a = [s.next() for _ in range(9)]
    s2 = RoundSchedule(CANON_TASKS, CANON_EMBODIMENTS, CANON_T1_HOSTS)
    s2.load(st)
    assert [s2.next() for _ in range(9)] == a
