import shutil
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


MAX_WORKERS = 3
AGENTS = ("bc", "gail", "bcq", "cql", "c2sac")
DATASETS = ("medium", "replay")
SEEDS = range(3)
ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Job:
    name: str
    checkpoint: Path
    command: list[str]


def build_jobs(extra_args: list[str]) -> list[Job]:
    jobs: list[Job] = []

    for agent in AGENTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                checkpoint = ROOT_DIR / "checkpoints" / agent / f"all-{dataset}" / f"seed-{seed}"
                command = [
                    sys.executable,
                    "trainer.py",
                    f"agent={agent}",
                    f"seed={seed}",
                    f"setting.dataset_name={dataset}",
                    *extra_args,
                ]
                jobs.append(Job(
                    name=f"{agent}:all:{dataset}:seed-{seed}",
                    checkpoint=checkpoint,
                    command=command,
                ))

    return jobs


def is_completed(checkpoint: Path) -> bool:
    model_path = checkpoint / "model.pth"
    log_path = checkpoint / "trainer.log"
    if not model_path.exists() or not log_path.exists():
        return False

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            return any(line.startswith("Test |") for line in handle)
    except OSError:
        return False


def prepare_job(job: Job) -> bool:
    if is_completed(job.checkpoint):
        print(f"[skip] {job.name} ({job.checkpoint.relative_to(ROOT_DIR)})", flush=True)
        return False

    if job.checkpoint.exists():
        print(f"[clean] {job.name} ({job.checkpoint.relative_to(ROOT_DIR)})", flush=True)
        shutil.rmtree(job.checkpoint)

    return True


def run_job(job: Job) -> tuple[Job, int]:
    print(f"[run ] {job.name}", flush=True)
    result = subprocess.run(job.command, cwd=ROOT_DIR)
    return job, result.returncode


def main() -> int:
    jobs = build_jobs(sys.argv[1:])

    pending_jobs: list[Job] = []
    skipped = 0
    for job in jobs:
        if not prepare_job(job):
            skipped += 1
        else:
            pending_jobs.append(job)

    print(
        f"Prepared {len(jobs)} jobs: {skipped} skipped, {len(pending_jobs)} queued, max_workers={MAX_WORKERS}",
        flush=True,
    )

    failures: list[tuple[Job, int]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, job) for job in pending_jobs]
        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode == 0:
                print(f"[done] {job.name}", flush=True)
            else:
                print(f"[fail] {job.name} (exit={returncode})", flush=True)
                failures.append((job, returncode))

    if failures:
        print(f"{len(failures)} job(s) failed.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
