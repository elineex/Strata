"""Tests for setup.py's UD-Q6_K_XL choice (docs/UNSLOTH_Q6.md): the six pinned shards with their sizes and
SHA-256, the RAM budget, the engine compiled into engine-q6k/ with the opt-in Q6_K kernels (never a downloaded
one), the refusals on AMD and on the CUDA 12 engine before any download, ensure_engine_for and the update sweep
visiting engine-q6k/.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_q6k
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

Q6 = "UD-Q6_K_XL"
REV = "38bb39ee97821de2c9009abb7e93950eec396e66"


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Pins(unittest.TestCase):
    def test_the_six_shards(self):
        fam = setup.FAMILIES["unsloth"]
        names = [setup.model_file(fam, Q6, i) for i in range(1, setup.model_shards(fam, Q6) + 1)]
        self.assertEqual(names, list(setup.UNSLOTH_Q6_SHARDS))
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_Q6_SHARDS.values()), 169165382688)   # docs/UNSLOTH_Q6.md
        for n, (size, sha) in setup.UNSLOTH_Q6_SHARDS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertEqual(fam["sha256"][n], (size, sha))
        self.assertEqual(fam["hf"].format(q=Q6),
                         f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-Q6_K_XL/")
        d = setup.MODELS[Q6]
        self.assertEqual((d["download_gb"], d["ram_gb"], d["arena_gb"]), (169.2, 48, 109.2))
        self.assertEqual(d["families"], ("unsloth",))
        self.assertTrue(d["experimental"] and d["budget"] and d["nvidia_only"] and d["q6k"])
        self.assertEqual(d["engine"], (0, 1, 40))                     # the first release with both halves
        self.assertEqual(d["doc"], "docs/UNSLOTH_Q6.md")
        self.assertNotIn("vision", d)                                 # no images (the family has none either)
        self.assertIn("quality", d["experimental_note"])              # and it says: not measured against llama.cpp

    def test_the_choice_from_a_file_name(self):
        fam = setup.FAMILIES["unsloth"]
        name = setup.model_file(fam, Q6, 1)
        self.assertIsNone(setup.gguf_unsupported(name))               # a file setup runs
        self.assertEqual(setup.gguf_choice(name), ("unsloth", Q6))
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/" + name]}), Q6)

    def test_budget(self):
        self.assertEqual(setup.resident_budget_gib(Q6, 63.7), 40)     # the 64 GB default, like UD-Q4_K_XL's
        self.assertEqual(setup.resident_budget_gib(Q6, 128), 101)     # at most all of them (109.2 GB = 101 GiB)
        self.assertAlmostEqual(setup.unsloth_split_need_gb(Q6), 169.2 + setup.UNSLOTH_RAM_LEFT_GB)


class FakeGGUF:
    """gguf_reader.GGUFFile: shard 3 holds the PLE table (docs/UNSLOTH_Q6.md: shard 3 is exactly the table)."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00003-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


class Base(unittest.TestCase):
    """setup.main() for the UD-Q6_K_XL choice, every outside effect mocked.  Unlike the other models, the engine
    is never ready-made: get_prebuilt must not be called and build_engine must get q6k=True."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified = [], [], []
        self.builds = []
        self.prebuilt = mock.Mock(return_value=self.t / "engine")

    def tearDown(self):
        self.tmp.cleanup()

    def fake_build_engine(self, gpu, vision, yes, llama, toolkit=None, q6k=False):
        eng = setup.engine_dir(toolkit, q6k)
        eng.mkdir(exist_ok=True)
        archs = sorted({int(x) for x in gpu.get("archs", [gpu["arch"]])})
        (eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.40", "archs": archs,
                                                    "vision": vision, **({"q6k": True} if q6k else {}),
                                                    "cuda_dirs": [str(self.t / "cudalib")]}))
        (eng / setup.EXE).write_bytes(b"engine")
        self.builds.append({"vision": vision, "toolkit": toolkit, "q6k": q6k})
        return eng

    def main(self, argv, ram=63.7, n_gpus=1, model=True, amd=(), free=500.0, m=Q6, answers=None, family=True):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others)."""
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        found = [{"index": i, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120", "driver": "580.97"}
                 for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "rotational_disk", lambda p: None),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "pip_cuda_libs", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", self.prebuilt),
            mock.patch.object(setup, "build_engine", self.fake_build_engine),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", "unsloth"] if family else []),
                                            *(["--model", m] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = self.t / f"strata-unsloth-{m.lower()}.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)


class Install(Base):
    def test_install(self):
        """The six shards, the RAM budget, and an engine compiled into engine-q6k/ - never a downloaded one."""
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.prebuilt.call_count, 0)                 # no ready-made engine has the Q6_K kernels
        self.assertEqual(self.builds, [{"vision": "none", "toolkit": 13, "q6k": True}])
        base = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-Q6_K_XL/"
        self.assertEqual(self.downloads, [base + n for n in setup.UNSLOTH_Q6_SHARDS])
        self.assertEqual(self.verified, [(n, *v) for n, v in setup.UNSLOTH_Q6_SHARDS.items()])
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertIn("--compat-bf16", packs[0])
        self.assertNotIn("--experts-bin", packs[0])
        self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith("UD-Q6_K_XL-00001-of-00006.gguf"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")
        self.assertTrue(args[args.index("--native") + 1].endswith("UD-Q6_K_XL-00001-of-00006.gguf"))
        for flag in ("--ple-gguf", "--mmap-experts", "--resident-experts", "--vision"):
            self.assertNotIn(flag, args)                              # six shards: the engine finds shard 3 by name
        self.assertEqual(Path(cfg["exe"]).parent.name, "engine-q6k")
        self.assertEqual(cfg["lib_dirs"], [str(self.t / "cudalib")])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-q6_k_xl")
        self.assertNotIn("vision", cfg)
        self.assertNotIn("cuda", cfg)                                 # CUDA 13: the key stays out, like the others
        with tempfile.TemporaryDirectory() as t:                      # a start reads the choice back
            p = Path(t) / "strata-unsloth-ud-q6_k_xl.json"
            p.write_text(json.dumps(cfg))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"]), ("unsloth", Q6))

    def test_experimental_warning_says_what_is_not_measured(self):
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        self.assertIn("UD-Q6_K_XL is EXPERIMENTAL (docs/UNSLOTH_Q6.md)", out)
        self.assertIn("has not been measured yet", out)               # not the Q4 claim of a checked quality
        self.assertNotIn("quality checked against llama.cpp", out)

    def test_amd_is_refused_before_the_download(self):
        """The HIP engine never compiles with STRATA_Q6K_EXPERTS: it refuses the file, so this is a stop, not the
        UD-Q4_K_XL ask-anyway - and it comes before the 169 GB download."""
        r9700 = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201",
                  "driver": "amdgpu"}]
        code, out, cfg = self.main(["--context", "8192", "--backend", "hip", "--vision", "no"], amd=r9700)
        self.assertEqual(code, 1, out)
        self.assertIn("Q6_K kernels are CUDA-only", out)
        self.assertNotIn("Try it anyway", out)                        # asked would only waste the download
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.builds, [])
        self.assertEqual(self.prebuilt.call_count, 0)

    def test_cuda12_is_refused_before_the_download(self):
        code, out, cfg = self.main(["--context", "8192", "--cuda", "12"])
        self.assertEqual(code, 1, out)
        self.assertIn("CUDA 12 engine does not have them", out)
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.builds, [])

    def test_the_menu_lists_it_last_and_installs_it(self):
        code, out, cfg = self.main(["--context", "8192", "--vision", "no"], model=False, family=False,
                                   answers={"Which model": "4", "Which size": "3"})
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-q6_k_xl")
        self.assertTrue(any("Which size" in q and "[1]" in q for q in self.asked), self.asked)   # still UD-IQ4_XS


class BuildEngine(unittest.TestCase):
    """build_engine(q6k=True): its own folders and the two opt-in kernels; the plain build stays as it was."""

    def build(self, q6k):
        root = Path(self.d)
        built = []

        def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
            built.append((target, defs, Path(bdir).name, bat))
            (bdir / setup.EXE).parent.mkdir(parents=True, exist_ok=True)
            (bdir / setup.EXE).write_bytes(b"engine")

        with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "cmake_build", cmake_build), \
                mock.patch.object(setup, "source_hash", lambda p: "S"), \
                mock.patch.object(setup, "install_build_tools",
                                  lambda gpu, yes: (str(root / "cuda" / "bin" / "nvcc"), None)), \
                mock.patch.object(setup, "source_version", lambda: "test"), \
                mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)):
            code, out = quiet(setup.build_engine, {"arch": "86"}, "none", True, "llama", 13, q6k)
        self.assertEqual(code, root / ("engine-q6k" if q6k else "engine"), out)
        return built, root

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_q6k_build(self):
        built, root = self.build(True)
        self.assertEqual([(t, f, b) for t, _, f, b in built],
                         [("strata", "build-q6k", "build-strata-q6k.bat")])   # its own build folder: the option
        self.assertIn("-DSTRATA_Q6K_EXPERTS=ON", built[0][1])                 # would flip build/'s CMakeCache
        self.assertIn("-DSTRATA_MMQ_KQUANTS=ON", built[0][1])
        meta = json.loads((root / "engine-q6k" / "BUILD.json").read_text())
        self.assertEqual((meta["source"], meta["q6k"], meta["archs"], meta["version"]),
                         ("local", True, [86], "test"))
        self.assertTrue((root / "engine-q6k" / setup.EXE).exists())
        self.assertFalse((root / "engine").exists())                          # the plain engine untouched
        self.assertFalse((root / "build").exists())

    def test_the_plain_build_is_unchanged(self):
        built, root = self.build(False)
        self.assertEqual([(t, f, b) for t, _, f, b in built], [("strata", "build", "build-strata.bat")])
        self.assertNotIn("-DSTRATA_Q6K_EXPERTS=ON", built[0][1])
        meta = json.loads((root / "engine" / "BUILD.json").read_text())
        self.assertNotIn("q6k", meta)


class EnsureEngine(unittest.TestCase):
    """A card added to a UD-Q6_K_XL install: recompiled into engine-q6k/; a Pascal / Volta card is refused (the
    CUDA 12 engine has no Q6_K build, so the use_cuda12 migration would leave a model that cannot load)."""

    RTX = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120"}
    RTX3090 = {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "arch": "86"}
    V100 = {"index": 2, "name": "NVIDIA Tesla V100", "vram_gb": 16.0, "arch": "70"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.eng = self.t / "engine-q6k"
        self.eng.mkdir()
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.40", "archs": [120],
                                                         "vision": "none", "q6k": True, "cuda_dirs": ["/cuda/lib"]}))
        (self.eng / setup.EXE).write_bytes(b"engine")
        self.cfg = {"exe": str(self.eng / setup.EXE), "args": ["--native", "x"], "lib_dirs": []}
        self.cfg_path = self.t / "strata-unsloth-ud-q6_k_xl.json"
        self.builds = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_build(self, gpu, vision, yes, llama, toolkit=None, q6k=False):
        self.builds.append({"vision": vision, "toolkit": toolkit, "q6k": q6k, "archs": gpu["archs"]})
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.40",
                                                         "archs": gpu["archs"], "vision": vision, "q6k": True,
                                                         "cuda_dirs": ["/cuda/lib"]}))
        return self.eng

    def ensure(self, cards):
        cfg = dict(self.cfg)
        with mock.patch.object(setup, "ROOT", self.t), \
                mock.patch.object(setup, "build_engine", self.fake_build), \
                mock.patch.object(setup, "gpu_info", lambda *a: dict(self.RTX)), \
                mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"):
            code, out = quiet(setup.ensure_engine_for, cards, self.cfg_path, cfg, True)
        return code, out, cfg

    def test_a_new_card_recompiles_engine_q6k(self):
        ret, out, cfg = self.ensure([self.RTX, self.RTX3090])
        self.assertEqual(ret["lib_dirs"], ["/cuda/lib"])
        self.assertEqual(self.builds, [{"vision": "none", "toolkit": 13, "q6k": True, "archs": [86, 120]}])
        self.assertIn("/cuda/lib", json.loads(self.cfg_path.read_text())["lib_dirs"])

    def test_the_installed_card_rebuilds_nothing(self):
        ret, out, cfg = self.ensure([self.RTX])
        self.assertEqual(ret["lib_dirs"], [])
        self.assertEqual(self.builds, [])

    def test_an_old_card_added_is_refused(self):
        code, out, cfg = self.ensure([self.RTX, self.V100])
        self.assertEqual(code, 1)
        self.assertIn("cannot take an old card", out)
        self.assertIn("--gpu N", out)
        self.assertEqual(self.builds, [])                              # not migrated to engine-cuda12/ either
        self.assertFalse(self.cfg_path.exists())


class UpdateSweep(unittest.TestCase):
    """setup on a PC that already has engine-q6k/: the sweep recompiles it when the source changed, and never
    treats it as ready-made (no ready-made engine has the Q6_K kernels)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.eng = self.t / "engine-q6k"
        self.eng.mkdir()
        (self.eng / setup.EXE).write_bytes(b"engine")
        self.builds = []
        self.pre = mock.Mock(return_value=None)

    def tearDown(self):
        self.tmp.cleanup()

    def fake_build(self, gpu, vision, yes, llama, toolkit=None, q6k=False):
        self.builds.append({"toolkit": toolkit, "q6k": q6k})
        return self.eng

    def sweep(self):
        with mock.patch.object(setup, "ROOT", self.t), mock.patch.object(setup, "build_engine", self.fake_build), \
                mock.patch.object(setup, "get_prebuilt", self.pre), \
                mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"), \
                mock.patch.object(setup, "gpu_info", lambda *a: {"index": 0, "name": "RTX 5070", "vram_gb": 11.9,
                                                                 "arch": "120"}), \
                mock.patch.object(setup, "amd_gpus", lambda *a: []), \
                mock.patch.object(setup, "source_hash", lambda p: "NEW"):
            return quiet(setup.update_installed_engine, "http://x")

    def test_a_local_engine_q6k_is_recompiled(self):
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.40", "archs": [120],
                                                         "vision": "none", "q6k": True, "src": "OLD"}))
        code, out = self.sweep()
        self.assertIsNone(code, out)
        self.assertEqual(self.builds, [{"toolkit": 13, "q6k": True}])
        self.assertEqual(self.pre.call_count, 0)

    def test_an_up_to_date_engine_q6k_is_left_alone(self):
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.40", "archs": [120],
                                                         "vision": "none", "q6k": True, "src": "NEW"}))
        code, out = self.sweep()
        self.assertIsNone(code, out)
        self.assertEqual(self.builds, [])

    def test_a_non_local_engine_q6k_is_never_downloaded(self):
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "prebuilt", "version": "0.1.40",
                                                         "archs": [120]}))
        code, out = self.sweep()
        self.assertIsNone(code, out)
        self.assertIn("no ready-made one has the Q6_K kernels", out)
        self.assertEqual(self.builds, [])
        self.assertEqual(self.pre.call_count, 0)


if __name__ == "__main__":
    unittest.main()
