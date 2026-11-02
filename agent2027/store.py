"""State outside the context window: `runs/{run_id}` and `tasks/{id}`.

Two backends with the same API: MemoryStore (one process, no setup) and
FirestoreStore (survives restarts, shared by notebook / Cloud Run / live view).
"""
import threading
import time
import uuid

from . import config


class MemoryStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._runs: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}

    # runs ---------------------------------------------------------------
    def update_run(self, run_id, **fields):
        with self._lock:
            self._runs.setdefault(run_id, {"findings": []}).update(fields, updated_at=time.time())

    def append_run(self, run_id, field, value):
        with self._lock:
            run = self._runs.setdefault(run_id, {"findings": []})
            run.setdefault(field, []).append(value)
            run["updated_at"] = time.time()

    def get_run(self, run_id) -> dict:
        with self._lock:
            return dict(self._runs.get(run_id, {}))

    # tasks --------------------------------------------------------------
    def add_task(self, run_id, role, prompt) -> str:
        task_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id, "run_id": run_id, "role": role, "prompt": prompt,
                "status": "pending", "worker_id": None, "result": None, "verified": False,
                "created_at": time.time(),
            }
        return task_id

    def tasks(self, run_id) -> list[dict]:
        with self._lock:
            return [dict(t) for t in self._tasks.values() if t["run_id"] == run_id]

    def claim_task(self, run_id, role, worker_id):
        """Atomically move one pending task for `role` to claimed."""
        with self._lock:
            for t in self._tasks.values():
                if t["run_id"] == run_id and t["role"] == role and t["status"] == "pending":
                    t.update(status="claimed", worker_id=worker_id)
                    return dict(t)
        return None

    def claim_review(self, run_id, worker_id):
        """Atomically claim one finished, unverified task for the fact checker."""
        with self._lock:
            for t in self._tasks.values():
                if (t["run_id"] == run_id and t["status"] == "done"
                        and not t["verified"] and not t.get("reviewer")):
                    t["reviewer"] = worker_id
                    return dict(t)
        return None

    def update_task(self, task_id, **fields):
        with self._lock:
            self._tasks[task_id].update(fields)


class FirestoreStore:
    def __init__(self, project=None):
        from google.cloud import firestore

        self._fs = firestore
        self.db = firestore.Client(project=project or config.PROJECT)

    # runs ---------------------------------------------------------------
    def _run_ref(self, run_id):
        return self.db.collection("runs").document(run_id)

    def update_run(self, run_id, **fields):
        self._run_ref(run_id).set({**fields, "updated_at": time.time()}, merge=True)

    def append_run(self, run_id, field, value):
        self._run_ref(run_id).set(
            {field: self._fs.ArrayUnion([value]), "updated_at": time.time()}, merge=True
        )

    def get_run(self, run_id) -> dict:
        snap = self._run_ref(run_id).get()
        return snap.to_dict() or {} if snap.exists else {}

    # tasks --------------------------------------------------------------
    def add_task(self, run_id, role, prompt) -> str:
        ref = self.db.collection("tasks").document()
        ref.set({
            "run_id": run_id, "role": role, "prompt": prompt, "status": "pending",
            "worker_id": None, "result": None, "verified": False, "created_at": time.time(),
        })
        return ref.id

    def tasks(self, run_id) -> list[dict]:
        out = []
        for d in self.db.collection("tasks").where("run_id", "==", run_id).stream():
            out.append({**d.to_dict(), "id": d.id})
        return out

    def _claim_in_txn(self, query, mutate):
        """Try candidates one by one; a transaction makes each claim race-safe."""
        firestore = self._fs

        @firestore.transactional
        def txn_fn(txn, ref):
            snap = ref.get(transaction=txn)
            data = snap.to_dict() or {}
            if not mutate["check"](data):
                return None
            txn.update(ref, mutate["fields"])
            return {**data, **mutate["fields"], "id": ref.id}

        for cand in query.limit(10).stream():
            claimed = txn_fn(self.db.transaction(), cand.reference)
            if claimed:
                return claimed
        return None

    def claim_task(self, run_id, role, worker_id):
        q = (self.db.collection("tasks").where("run_id", "==", run_id)
             .where("role", "==", role).where("status", "==", "pending"))
        return self._claim_in_txn(q, {
            "check": lambda d: d.get("status") == "pending",
            "fields": {"status": "claimed", "worker_id": worker_id},
        })

    def claim_review(self, run_id, worker_id):
        q = (self.db.collection("tasks").where("run_id", "==", run_id)
             .where("status", "==", "done").where("verified", "==", False))
        return self._claim_in_txn(q, {
            "check": lambda d: d.get("status") == "done" and not d.get("verified") and not d.get("reviewer"),
            "fields": {"reviewer": worker_id},
        })

    def update_task(self, task_id, **fields):
        self.db.collection("tasks").document(task_id).update(fields)


_store = None
_store_lock = threading.Lock()


def get_store():
    """One shared store per process (memory workers must share the same object)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = FirestoreStore() if config.STORE == "firestore" else MemoryStore()
        return _store


def set_store(store):
    global _store
    _store = store
