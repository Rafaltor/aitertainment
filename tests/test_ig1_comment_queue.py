"""Tests ig1_comment_queue."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ig1_comment_queue as q


class Ig1CommentQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue = Path(self.tmp.name) / "queue.json"
        self.commented = Path(self.tmp.name) / "commented.json"

    def test_enqueue_and_pop_fifo(self) -> None:
        self.assertTrue(
            q.enqueue_ig1_comment("ABC", username="u1", path=self.queue)
        )
        self.assertTrue(
            q.enqueue_ig1_comment("DEF", username="u2", path=self.queue)
        )
        self.assertFalse(
            q.enqueue_ig1_comment("ABC", username="u1", path=self.queue)
        )
        first = q.pop_ig1_comment(path=self.queue)
        second = q.pop_ig1_comment(path=self.queue)
        self.assertEqual(first["media_id"], "ABC")
        self.assertEqual(second["media_id"], "DEF")
        self.assertIsNone(q.pop_ig1_comment(path=self.queue))

    def test_mark_commented_blocks_reenqueue(self) -> None:
        old_c, old_q = q.DEFAULT_COMMENTED_PATH, q.DEFAULT_QUEUE_PATH
        q.DEFAULT_COMMENTED_PATH = self.commented
        q.DEFAULT_QUEUE_PATH = self.queue
        self.addCleanup(setattr, q, "DEFAULT_COMMENTED_PATH", old_c)
        self.addCleanup(setattr, q, "DEFAULT_QUEUE_PATH", old_q)
        q.mark_commented("XYZ")
        self.assertFalse(q.enqueue_ig1_comment("XYZ"))

    def test_prune_old_commented(self) -> None:
        old = time.time() - q.COMMENTED_TTL_S - 10
        data = {"reels": {"OLD": old, "NEW": time.time()}}
        self.commented.write_text(json.dumps(data), encoding="utf-8")
        self.assertFalse(q.was_recently_commented("OLD", path=self.commented))
        self.assertTrue(q.was_recently_commented("NEW", path=self.commented))


if __name__ == "__main__":
    unittest.main()
