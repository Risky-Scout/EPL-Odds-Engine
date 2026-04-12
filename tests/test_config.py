from pathlib import Path
import tempfile
import unittest

from epl_pmf_backtest.config import ProjectConfig


class TestConfig(unittest.TestCase):
    def test_storage_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            path.write_text(
                "project:\n  league: ENG Premier League\nstorage:\n  root_dir: ./workspace/epl_joint_pmf\n",
                encoding="utf-8",
            )
            cfg = ProjectConfig.from_yaml(path)
            self.assertEqual(cfg.league, "ENG Premier League")
            self.assertTrue(str(cfg.storage_root).endswith("workspace/epl_joint_pmf"))


if __name__ == "__main__":
    unittest.main()
