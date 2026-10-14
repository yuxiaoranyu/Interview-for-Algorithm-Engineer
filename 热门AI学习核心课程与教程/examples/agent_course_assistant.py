"""Standard-library control-layer lab; no LLM, retrieval service, or HTTP auth.

Run this file with Python 3.10+. Database files live in a temporary directory.
The user_id parameters represent an already authenticated caller.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import sqlite3
import tempfile
import uuid


def rrf(rankings, constant=60):
    """Fuse ranked lists of string IDs; each list gives each ID at most one vote."""
    if isinstance(constant, bool) or not isinstance(constant, (int, float)):
        raise ValueError('constant must be numeric')
    if not math.isfinite(constant) or constant <= 0:
        raise ValueError('constant must be positive and finite')
    scores = defaultdict(float)
    for ranking in rankings:
        unique = list(dict.fromkeys(ranking))
        if any(not isinstance(item, str) for item in unique):
            raise ValueError('document IDs must be strings')
        for rank, item in enumerate(unique, 1):
            scores[item] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Memory:
    """In-memory exercise: explicit writes, user isolation, revision and erasure."""
    def __init__(self):
        self.records = []

    def read(self, user_id, key):
        matches = [r for r in self.records if r['user_id'] == user_id
                   and r['key'] == key and r['status'] == 'Active']
        return dict(matches[-1]) if matches else None

    def write(self, user_id, key, value, *, evidence, explicit):
        if explicit is not True or not evidence or not isinstance(value, str):
            raise ValueError('an explicit statement and its evidence are required')
        old = [r for r in self.records if r['user_id'] == user_id and r['key'] == key]
        version = 1 + max((r['version'] for r in old), default=0)
        for record in old:
            if record['status'] == 'Active':
                record['status'] = 'Outdated'
        self.records.append(dict(user_id=user_id, key=key, value=value,
                                 evidence=evidence, version=version, status='Active'))

    def delete(self, user_id, key):
        for record in self.records:
            if record['user_id'] == user_id and record['key'] == key:
                record.update(status='Deleted', value=None, evidence=None)


@dataclass(frozen=True)
class Proposal:
    user_id: str
    slot_id: str
    version: int

    def canonical(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))


class Booking:
    """One free seat per user per slot; local transactions, not distributed exactly-once."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS slots(
                slot_id TEXT PRIMARY KEY, remaining INTEGER NOT NULL CHECK(remaining >= 0),
                version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS confirmations(
                token TEXT PRIMARY KEY, user_id TEXT NOT NULL, proposal TEXT NOT NULL,
                expires REAL NOT NULL, used_key TEXT);
            CREATE TABLE IF NOT EXISTS bookings(
                idem_key TEXT PRIMARY KEY, user_id TEXT NOT NULL, slot_id TEXT NOT NULL,
                proposal TEXT NOT NULL, booking_id TEXT UNIQUE NOT NULL,
                UNIQUE(user_id, slot_id));
        ''')

    def seed(self):
        self.db.execute("INSERT OR IGNORE INTO slots VALUES ('python-evening', 2, 1)")

    def prepare(self, user_id, slot_id):
        row = self.db.execute('SELECT remaining, version FROM slots WHERE slot_id=?',
                              (slot_id,)).fetchone()
        if not row or row[0] <= 0:
            raise ValueError('slot unavailable')
        return Proposal(user_id, slot_id, row[1])

    def confirm(self, user_id, proposal, *, now, ttl=30):
        # Called only by a trusted handler after an explicit user confirmation.
        if (proposal.user_id != user_id or not math.isfinite(now)
                or isinstance(ttl, bool) or not math.isfinite(ttl) or ttl <= 0):
            raise ValueError('invalid confirmation')
        token = uuid.uuid4().hex
        self.db.execute('INSERT INTO confirmations VALUES (?, ?, ?, ?, NULL)',
                        (token, user_id, proposal.canonical(), now + ttl))
        return token

    def commit(self, user_id, idem_key, proposal, token, *, now):
        if not idem_key or proposal.user_id != user_id or not math.isfinite(now):
            raise ValueError('invalid caller or request')
        payload = proposal.canonical()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT user_id, proposal, booking_id FROM bookings '
                                    'WHERE idem_key=?', (idem_key,)).fetchone()
            if prior:
                if prior[:2] != (user_id, payload):
                    raise ValueError('idempotency key reused with different identity or payload')
                self.db.execute('COMMIT')
                return prior[2]
            permission = self.db.execute('SELECT user_id, proposal, expires, used_key '
                                         'FROM confirmations WHERE token=?', (token,)).fetchone()
            if (not permission or permission[:2] != (user_id, payload)
                    or permission[2] <= now or permission[3] is not None):
                raise ValueError('confirmation missing, mismatched, expired, or already used')
            changed = self.db.execute('UPDATE slots SET remaining=remaining-1, version=version+1 '
                                      'WHERE slot_id=? AND version=? AND remaining>0',
                                      (proposal.slot_id, proposal.version)).rowcount
            if changed != 1:
                raise ValueError('availability changed; prepare and confirm again')
            booking_id = uuid.uuid4().hex
            self.db.execute('INSERT INTO bookings VALUES (?, ?, ?, ?, ?)',
                            (idem_key, user_id, proposal.slot_id, payload, booking_id))
            self.db.execute('UPDATE confirmations SET used_key=? WHERE token=?', (idem_key, token))
            self.db.execute('COMMIT')
            return booking_id
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            raise

    def close(self):
        self.db.close()


def run_lab():
    checks = []

    def rejected(label, action):
        try:
            action()
        except (ValueError, sqlite3.IntegrityError):
            checks.append(label)
        else:
            raise AssertionError(label + ' was unexpectedly accepted')

    fused = rrf([['a', 'b', 'a'], ['b', 'c']])
    assert [x[0] for x in fused] == ['b', 'a', 'c']
    assert math.isclose(fused[0][1], 1 / 62 + 1 / 61)
    assert rrf([[], []]) == []
    assert rrf([['b'], ['a']]) == [('a', 1/61), ('b', 1/61)]
    checks.append('rrf_rank_fusion_dedup_empty_tie')

    memory = Memory()
    memory.write('u1', 'time', 'evening', evidence='message-1', explicit=True)
    snapshot = json.dumps(memory.records, sort_keys=True)
    value = memory.read('u1', 'time')
    assert value['value'] == 'evening' and memory.read('u2', 'time') is None
    value['value'] = 'mutated-copy'
    assert json.dumps(memory.records, sort_keys=True) == snapshot
    checks.append('memory_read_has_no_write_and_is_user_scoped')
    rejected('memory_question_cannot_write', lambda: memory.write(
        'u1', 'time', 'morning', evidence='what is my preference?', explicit=False))
    memory.write('u1', 'time', 'morning', evidence='message-2', explicit=True)
    assert memory.records[0]['status'] == 'Outdated' and memory.read('u1', 'time')['version'] == 2
    memory.delete('u1', 'time')
    assert memory.read('u1', 'time') is None and all(r['value'] is None for r in memory.records)
    checks.append('memory_revision_and_content_erasure')

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'booking.sqlite'
        b = Booking(path)
        b.seed()
        p = b.prepare('u1', 'python-evening')
        other = b.prepare('u2', 'python-evening')
        token = b.confirm('u1', p, now=100)
        other_token = b.confirm('u2', other, now=100)
        rejected('changed_payload_requires_new_confirmation', lambda: b.commit(
            'u1', 'k1', Proposal('u1', 'another-slot', 1), token, now=101))
        rejected('wrong_user_cannot_commit', lambda: b.commit('u2', 'k1', p, token, now=101))
        rejected('confirmation_expires', lambda: b.commit('u1', 'k1', p, token, now=131))
        result = b.commit('u1', 'k1', p, token, now=101)
        b.close()  # Simulate a lost response followed by a fresh connection.
        b = Booking(path)
        assert b.commit('u1', 'k1', p, token, now=200) == result
        assert b.db.execute('SELECT COUNT(*) FROM bookings').fetchone()[0] == 1
        assert b.db.execute('SELECT remaining FROM slots').fetchone()[0] == 1
        checks.append('lost_response_retry_returns_same_booking_without_second_decrement')
        rejected('idempotency_payload_mismatch', lambda: b.commit(
            'u1', 'k1', Proposal('u1', 'python-evening', 2), token, now=102))
        rejected('confirmation_cannot_create_second_intent', lambda: b.commit(
            'u1', 'new-key', p, token, now=102))
        rejected('stale_availability_requires_reconfirmation', lambda: b.commit(
            'u2', 'k2', other, other_token, now=102))
        refreshed = b.prepare('u2', 'python-evening')
        refreshed_token = b.confirm('u2', refreshed, now=103)
        b.commit('u2', 'k2', refreshed, refreshed_token, now=104)
        rejected('sold_out', lambda: b.prepare('u3', 'python-evening'))
        assert b.db.execute('SELECT COUNT(*) FROM bookings').fetchone()[0] == 2
        assert b.db.execute('SELECT remaining FROM slots').fetchone()[0] == 0
        checks.append('transaction_rollbacks_preserve_capacity')
        b.close()
    return dict(status='passed', checks=checks, scope='local deterministic control-layer exercise')


if __name__ == '__main__':
    print(json.dumps(run_lab(), ensure_ascii=False, indent=2))
