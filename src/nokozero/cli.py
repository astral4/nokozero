"""Command-line interface for the nokozero environment driver."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nokozero import catalog
from nokozero.env import Env, EnvConfig
from nokozero.replay import harvest, ingest, rpy, synth
from nokozero.spawn import default_hook_dll, hook_image
from nokozero.train.config import TrainConfig
from nokozero.train.curriculum import CurriculumConfig, build_pool
from nokozero.train.features import ACTION_SETS, VARIANTS, Featurizer
from nokozero.train.rollout import (
    FULL_STAGE_MAX_STEPS,
    GAME_START_POWER,
    SEGMENT_MAX_STEPS,
    TRAINING_POWER,
    parse_spec,
    parse_target,
)
from nokozero.utils import atomic_write, check_writable, code_version, keep_source, print_row

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from _typeshed import DataclassInstance

    from nokozero.train.game import GameResult, StagePolicy, StageTrace


def _config_flags(config: type[DataclassInstance], args: argparse.Namespace) -> dict[str, Any]:
    """Return the flags in `args` specifying fields of `config`."""
    given = vars(args)
    return {field.name: given[field.name] for field in fields(config) if field.name in given}


def _env_config(args: argparse.Namespace) -> EnvConfig:
    """Build the fleet config from the flags that `command` gives to every subcommand."""
    return EnvConfig(**_config_flags(EnvConfig, args))


def _train_config(args: argparse.Namespace) -> TrainConfig:
    """Build the run's config from the `train` flags."""
    flags = _config_flags(TrainConfig, args)
    gamma, n_step = TrainConfig.horizon_for(args.step_interval)
    flags.setdefault("gamma", gamma)
    flags.setdefault("n_step", n_step)
    if "max_minutes" in args:
        flags["max_seconds"] = args.max_minutes * 60
    return TrainConfig(
        **flags,
        sections=tuple(args.section),
        curriculum=CurriculumConfig(**_config_flags(CurriculumConfig, args)),
    )


def _train(args: argparse.Namespace) -> int:
    from nokozero.train.run import train  # noqa: PLC0415

    train(_train_config(args), _env_config(args))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from nokozero.train import evaluate  # noqa: PLC0415

    evaluation = evaluate.Evaluation(
        args.section if args.target is None else args.target,
        prefix_router=args.prefix_router,
        full_stage=args.full_stage,
        until_tic=args.until_tic,
        max_episode_steps=args.max_episode_steps,
        start_power=args.start_power,
        dither=args.dither,
        noise=args.noise,
    )
    seeds = evaluate.sample_seeds(args.seeds)
    evaluate.evaluate(args.model, evaluation, seeds, _env_config(args), out=args.out)
    return 0


def _results(args: argparse.Namespace) -> int:
    from nokozero.train.results import Result, segment  # noqa: PLC0415  # no JAX: fast

    for path in args.results:
        result = Result.load(path)
        if args.segment is None:
            print_row({"path": str(path), **result.row()})
            continue
        key = str(args.segment)
        by_seed = result.segments.get(key, {})
        print_row({"path": str(path), "segment": key, **segment(by_seed)})
    return 0


def _stages(args: argparse.Namespace, first: int, seeds: Iterable[int]) -> tuple[dict[int, StagePolicy], int]:
    """Load `--routers` from stage `first` onward. Return the stages and the character used for their policies."""
    from nokozero.train import game  # noqa: PLC0415

    stages = {first + i: game.StagePolicy.load(spec) for i, spec in enumerate(args.routers)}
    return stages, game.check_runs(stages, seeds, start_power=args.start_power)


def _play(
    args: argparse.Namespace,
    stages: Mapping[int, StagePolicy],
    seeds: Iterable[int],
    traces: dict[int, list[StageTrace]] | None = None,
) -> GameResult:
    """Play the runs of `game` and `synthesize` on a fleet booted for them."""
    from nokozero.train import game  # noqa: PLC0415

    return game.play_game(
        stages,
        seeds,
        _env_config(args),
        start_power=args.start_power,
        max_steps=args.max_episode_steps,
        log=print_row,
        traces=traces,
    )


def _game(args: argparse.Namespace) -> int:
    from nokozero.train import evaluate, game  # noqa: PLC0415

    check_writable(args.out)
    seeds = evaluate.sample_seeds(args.seeds)
    stages, _ = _stages(args, args.first_stage, seeds)
    game.report(_play(args, stages, seeds), args.out)
    return 0


def _synthesize(args: argparse.Namespace) -> int:
    if len(args.routers) != len(synth.STAGES):
        msg = f"--routers takes one spec per stage, {synth.STAGES[0]} to {synth.STAGES[-1]}, not {len(args.routers)}"
        raise ValueError(msg)
    stages, character = _stages(args, synth.STAGES[0], args.seeds)
    template = synth.Template(
        rpy.load(args.template),
        character=character,
        difficulty=catalog.difficulty_for(catalog.opening(synth.STAGES[0])),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[int, list[StageTrace]] = {}
    result = _play(args, stages, args.seeds, traces)
    written = unwritten = 0
    for seed, result in sorted(result.per_seed.items()):
        if result != "clear":
            print(f"seed {seed}: {result}")  # noqa: T201
            continue
        try:
            data = synth.synthesize(traces[seed], template, name=args.name)
        except synth.SynthError as e:
            print(f"seed {seed}: clear, not written: {e}", file=sys.stderr)  # noqa: T201
            unwritten += 1
            continue
        path = args.out_dir / f"{args.name}-{seed}.rpy"
        atomic_write(path, data)
        written += 1
        print(f"seed {seed}: clear -> {path} ({len(data)} bytes)")  # noqa: T201
    refused = f"; {unwritten} cleared and were not" if unwritten else ""
    print(f"{written} of {len(result.per_seed)} runs cleared and were written{refused}")  # noqa: T201
    return 1 if unwritten else 0


def _ingest(args: argparse.Namespace) -> int:
    store = ingest.Store(args.out)
    done = store.keys()
    paths = sorted(args.replays.glob("*.rpy"))
    if args.limit is not None:
        paths = paths[: args.limit]
    jobs = ingest.jobs_for(paths, character=args.character, difficulty=args.difficulty, skip=done.__contains__)
    with Env(_env_config(args)) as env:
        verified = ingest.run(env, jobs, store)
    print(f"verified {verified} stage-plays; index at {store.index}")  # noqa: T201
    return 0


def _harvest(args: argparse.Namespace) -> int:
    check_writable(args.out)
    store = ingest.Store(args.store)
    pool = build_pool(store, args.target, args.character)
    if not pool:
        msg = f"no verified segments {args.target} in {args.store}"
        raise ValueError(msg)
    featurizer = Featurizer(
        k=args.k,
        variant=args.features,
        step_interval=args.step_interval,
        character=args.character,
    )
    demonstrations: list[harvest.Demonstration] = []
    with Env(_env_config(args)) as env:
        harvester = harvest.Harvester(env, iter(pool), featurizer)
        while not harvester.done:
            demonstrations.extend(harvester.step())
    if not demonstrations:
        msg = (
            f"none of the {len(pool)} segments survived playback ({harvester.desynced} desynced, "
            f"{harvester.unreachable} past a tape's reach); nothing to write to {args.out}"
        )
        raise ValueError(msg)
    harvest.save(args.out, demonstrations, featurizer)
    states = sum(len(d.features) for d in demonstrations)
    print(  # noqa: T201
        f"harvested {len(demonstrations)} of {len(pool)} segments ({states} states, {harvester.desynced} desynced, "
        f"{harvester.unreachable} past a tape's reach) to {args.out}"
    )
    return 0


def _bc(args: argparse.Namespace) -> int:
    from nokozero.train import imitation  # noqa: PLC0415

    check_writable(args.out)
    demos = imitation.Demonstrations.load(args.data)
    actions = ACTION_SETS[args.actions]

    policy = imitation.train(
        demos,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        log=print_row,
    )
    imitation.save(policy, args.out)
    return 0


def _argument[T](parse: Callable[[str], T]) -> Callable[[str], T]:
    """Wrap `parse` as an argparse `type`."""

    def convert(text: str) -> T:
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return convert


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(prog="nokozero")
    commands = parser.add_subparsers(required=True)

    def command(  # noqa: PLR0913
        name: str,
        help_text: str,
        run: Callable[[argparse.Namespace], int],
        *,
        character: bool = True,
        collects: bool = False,
        omit_unset: bool = False,
    ) -> argparse.ArgumentParser:
        """Add a subcommand that boots fleets."""
        sub = commands.add_parser(name, help=help_text, argument_default=argparse.SUPPRESS if omit_unset else None)
        sub.set_defaults(run=run)
        sub.add_argument("-d", "--game-dir", type=Path, required=True)
        sub.add_argument("--hook-dll", type=Path, default=default_hook_dll())
        sub.add_argument("--headed", action="store_true")
        if character:
            sub.add_argument(
                "--character",
                type=int,
                default=EnvConfig.character,
                help="character to play; chosen at boot and fixed for the fleet's lifetime",
            )
        sub.add_argument(
            "--instances",
            dest="num_instances",
            type=int,
            default=EnvConfig.num_instances,
            help="run every command on this many instances at once, in lockstep",
        )
        sub.add_argument(
            "--step-timeout",
            type=float,
            default=EnvConfig.step_timeout,
            help="number of seconds that an instance can take to answer a command before it is declared dead"
        )
        sub.add_argument(
            "--startup-timeout",
            type=float,
            default=EnvConfig.startup_timeout,
            help="number of seconds from spawn to an instance's first observation (raise under load)",
        )
        if collects:
            sub.add_argument(
                "--collect-window",
                type=float,
                default=EnvConfig.collect_window,
                help="number of seconds to keep collecting a step's late answers into the next decision, "
                "capped by the decision's own time",
            )
        return sub

    def featurizer_flags(sub: argparse.ArgumentParser) -> None:
        """Add the flags that build a featurizer (`Featurizer`)."""
        sub.add_argument("--k", type=int, default=TrainConfig.k, help="featurizer token budget")
        sub.add_argument(
            "--features",
            choices=VARIANTS,
            default=TrainConfig.features,
            help="the feature set",
        )
        sub.add_argument(
            "--step-interval",
            type=int,
            default=TrainConfig.step_interval,
            help="game frames per decision (1-60)",
        )

    def actions_flag(sub: argparse.ArgumentParser) -> None:
        """Add the action-set flag of the commands that build a policy."""
        sub.add_argument(
            "--actions",
            choices=sorted(ACTION_SETS),
            default=TrainConfig.actions,
            help="action set: held directions, or holds plus 1- and 2-frame taps (tap3)",
        )

    def play_flags(sub: argparse.ArgumentParser, routers_help: str) -> None:
        """Add the flags of the commands that play entire games."""
        sub.add_argument("--routers", type=Path, nargs="+", required=True, help=routers_help)
        sub.add_argument(
            "--start-power",
            type=int,
            default=GAME_START_POWER,
            help="the first stage's starting power * 100",
        )
        sub.add_argument(
            "--max-episode-steps",
            type=int,
            default=FULL_STAGE_MAX_STEPS,
            help="decisions before a stage's run is truncated",
        )

    train_cmd = command(
        "train",
        "train a segment's expert from its start sources",
        _train,
        collects=True,
        omit_unset=True,
    )
    train_cmd.add_argument(
        "--target",
        type=_argument(parse_target),
        help="the trained segment: opening>word, or opening>word#n for a word's n-th entry "
        "(reached by the replay and prefix starts)",
    )
    train_cmd.add_argument(
        "--section",
        nargs="*",
        type=_argument(parse_spec),
        default=[],
        help="warp section spec(s) for playing runs",
    )
    train_cmd.add_argument(
        "--replay-weight",
        type=int,
        help="the store's replay starts per rotation of the start sources",
    )
    train_cmd.add_argument(
        "--prefix-weight",
        type=int,
        help="the prefix router's starts per rotation of the start sources",
    )
    train_cmd.add_argument(
        "--warp-weight",
        type=int,
        help="the warp sections' starts per rotation of the start sources",
    )
    train_cmd.add_argument("--out", type=Path, required=True, help="directory for logs and checkpoints")
    train_cmd.add_argument("--seed", type=int)
    featurizer_flags(train_cmd)
    train_cmd.add_argument("--total-steps", type=int)
    train_cmd.add_argument("--warmup-steps", type=int)
    train_cmd.add_argument("--max-episode-steps", type=int)
    train_cmd.add_argument("--capacity", type=int)
    train_cmd.add_argument(
        "--archive-capacity",
        type=int,
        help="rows of a long-term archive tier that subsamples the whole run (0: none)",
    )
    train_cmd.add_argument("--batch-size", type=int)
    train_cmd.add_argument("--model-dim", type=int)
    train_cmd.add_argument("--model-heads", type=int)
    train_cmd.add_argument("--model-depth", type=int)
    train_cmd.add_argument(
        "--update-every",
        type=int,
        help="stored transitions between gradient steps (the replay ratio's denominator)",
    )
    train_cmd.add_argument(
        "--gamma",
        type=float,
        help="survival discount per decision; defaults to a 10 s horizon at any interval",
    )
    train_cmd.add_argument("--n-step", type=int, help="n-step window in decisions; defaults to 15 frames")
    train_cmd.add_argument(
        "--damage-bonus",
        type=float,
        help="add this much of the boss HP ratio a window takes to its target, capped at 1 "
        "(see `TrainConfig.damage_bonus`)",
    )
    train_cmd.add_argument(
        "--start-power",
        type=int,
        help="the power amount * 100 that seed and prefix starts warp in with",
    )
    train_cmd.add_argument(
        "--start-power-max",
        type=int,
        help="draw each seed or prefix start's power uniformly from --start-power to this "
        "(see `TrainConfig.start_power_max`)",
    )
    train_cmd.add_argument("--max-minutes", type=float, help="stop after this much wall-clock time")
    actions_flag(train_cmd)
    train_cmd.add_argument(
        "--uniform-starts",
        dest="uniform",
        action="store_true",
        help="draw curriculum starts uniformly over whole segments from the first episode",
    )
    train_cmd.add_argument("--explore-with", type=Path, help="a cloned policy (`bc`) to explore with")
    train_cmd.add_argument("--init-from", type=Path, help="continue training from this checkpoint's weights")
    train_cmd.add_argument(
        "--until-tic",
        type=int,
        help="end every episode as complete at this many frames into the chapter",
    )
    train_cmd.add_argument(
        "--demos",
        type=Path,
        help="a harvest whose segments enter the replay buffer as completed episodes "
        "before the first update (see `TrainConfig.demos`)",
    )
    train_cmd.add_argument(
        "--suffix",
        type=Path,
        help="the next segment's expert (a checkpoint or a router spec) that plays on from "
        "every completed episode's end state; its outcome is the terminal value (see `TrainConfig.suffix`)",
    )
    train_cmd.add_argument(
        "--handoff",
        type=Path,
        help="the next segment's expert; its value of a completed episode's end state "
        "replaces the terminal value 1 (see `TrainConfig.handoff`)",
    )
    train_cmd.add_argument(
        "--prefix-router",
        type=Path,
        help="router spec whose policy plays from the target's stage opening to the target",
    )
    train_cmd.add_argument("--seed-pool", type=int, help="draw seeds from a fixed pool of this many (prefix reuse)")
    train_cmd.add_argument(
        "--start-jitter",
        action="store_true",
        help="draw each seed start's landing position from the bottom of the playfield at random",
    )
    train_cmd.add_argument(
        "--buffer-seed-fraction",
        type=float,
        help="the share of every batch drawn from seed episodes",
    )
    train_cmd.add_argument("--epsilon-decay-steps", type=int)
    train_cmd.add_argument("--epsilon-start", type=float)
    train_cmd.add_argument("--epsilon-end", type=float)
    train_cmd.add_argument("--learning-rate", type=float)
    train_cmd.add_argument("--tau", type=float, help="Polyak rate of the target network")
    train_cmd.add_argument(
        "--decay-updates",
        type=int,
        help="decay the learning rate (lr) linearly to 0.1 * lr over this many gradient steps",
    )
    train_cmd.add_argument("--log-every", type=float)
    train_cmd.add_argument(
        "--store",
        type=Path,
        help="replay store to draw the target's replay starts from",
    )
    train_cmd.add_argument(
        "--initial-depth",
        type=int,
        help="number of frames before segment ending to draw the first curriculum starts from",
    )
    train_cmd.add_argument(
        "--curriculum-threshold",
        dest="threshold",
        type=float,
        help="completion rate at which curriculum starts move earlier",
    )

    eval_cmd = command(
        "evaluate",
        "play a section greedily from saved models over seeds",
        _evaluate,
        character=False,
        collects=True,
    )
    eval_cmd.add_argument(
        "--model",
        type=Path,
        nargs="+",
        required=True,
        help="checkpoints (.eqx) or router specs (.json) played in turn on a fleet",
    )
    where = eval_cmd.add_mutually_exclusive_group()
    where.add_argument(
        "--section",
        type=_argument(parse_spec),
        default="1202",
        help="section spec (section, or section@0 for the landing)",
    )
    where.add_argument(
        "--target",
        type=_argument(parse_target),
        help="with --prefix-router, the segment reached by the prefix (opening>word, or opening>word#n)",
    )
    eval_cmd.add_argument("--seeds", type=int, default=256, help="distinct seeds to evaluate")
    eval_cmd.add_argument(
        "--dither",
        type=float,
        default=0.0,
        help="draw each action among those within this much of the best score (seeded)",
    )
    eval_cmd.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian noise added to every action score before the choice (seeded)",
    )
    eval_cmd.add_argument(
        "--max-episode-steps",
        type=int,
        help=f"decisions before an episode is truncated (default {SEGMENT_MAX_STEPS}, as in training, "
        f"or {FULL_STAGE_MAX_STEPS} with --full-stage)",
    )
    eval_cmd.add_argument(
        "--out",
        type=Path,
        help="output directory with model and seed results",
    )
    eval_cmd.add_argument(
        "--until-tic",
        type=int,
        help="count an episode as complete at this many frames into the chapter",
    )
    eval_cmd.add_argument(
        "--full-stage",
        action="store_true",
        help="play from the section's landing through every chapter until a hit or the stage ends",
    )
    eval_cmd.add_argument(
        "--prefix-router",
        type=Path,
        help="router spec whose policy plays from --target's stage opening to the target",
    )
    eval_cmd.add_argument(
        "--start-power",
        type=int,
        default=TRAINING_POWER,
        help="the power amount * 100 that the landing warps in with",
    )
    game_cmd = command(
        "game",
        "play entire games by running the stage routers in order",
        _game,
        character=False,
        collects=True,
    )
    play_flags(game_cmd, "router specs (JSON) in stage order")
    game_cmd.add_argument("--first-stage", type=int, default=1, help="the stage played by the first router")
    game_cmd.add_argument("--seeds", type=int, default=256, help="distinct run seeds to play")
    game_cmd.add_argument("--out", type=Path, help="write the result as JSON")

    synth_cmd = command(
        "synthesize",
        "play seeds through the stage routers and write each clear as a .rpy replay",
        _synthesize,
        character=False,
        collects=True,
    )
    play_flags(
        synth_cmd,
        "one router spec per stage from stages 1 to 6",
    )
    synth_cmd.add_argument("--seeds", type=int, nargs="+", required=True, help="run seeds to play")
    synth_cmd.add_argument("--template", type=Path, required=True, help="a played .rpy of the same character and rank")
    synth_cmd.add_argument("--out-dir", type=Path, required=True, help="where the .rpy files go")
    synth_cmd.add_argument("--name", default=synth.NAME, help="the replay's player name (8 chars)")
    ingest_cmd = command("ingest", "play a replay corpus stage by stage, verify it, and label its segments", _ingest)
    ingest_cmd.add_argument("--replays", type=Path, required=True, help="directory of .rpy files to play")
    ingest_cmd.add_argument("--out", type=Path, required=True, help="store directory (traces/ and stages.ndjson)")
    ingest_cmd.add_argument(
        "--difficulty",
        type=int,
        default=catalog.LUNATIC,
        help="only play stages recorded at this difficulty (0-3; 3 is Lunatic)",
    )
    ingest_cmd.add_argument("--limit", type=int, help="play only the first N files in name order (for trials)")
    harvest_cmd = command("harvest", "featurize a chapter's clean segments for behavior cloning", _harvest)
    harvest_cmd.add_argument("--store", type=Path, required=True, help="an `ingest` store")
    harvest_cmd.add_argument(
        "--target",
        type=_argument(parse_target),
        required=True,
        help="the segment to harvest (opening>word, or opening>word#n)",
    )
    featurizer_flags(harvest_cmd)
    harvest_cmd.add_argument("--out", type=Path, required=True, help="the .npz to write")

    results_cmd = commands.add_parser("results", help="print evaluation results")
    results_cmd.set_defaults(run=_results)
    results_cmd.add_argument("results", type=Path, nargs="+", help="`evaluate` result files")
    results_cmd.add_argument(
        "--segment",
        type=_argument(parse_target),
        help="a full stage's segment (opening>word, or opening>word#n)",
    )

    bc_cmd = commands.add_parser("bc", help="clone a policy from harvested demonstrations")
    bc_cmd.set_defaults(run=_bc)
    bc_cmd.add_argument("--data", type=Path, required=True, help="a `harvest` .npz")
    bc_cmd.add_argument("--out", type=Path, required=True, help="the policy file to write")
    actions_flag(bc_cmd)
    bc_cmd.add_argument("--epochs", type=int, default=10)
    bc_cmd.add_argument("--batch-size", type=int, default=256)
    bc_cmd.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `nokozero` command."""
    args = _parse_args(argv)
    try:
        hook_dll: Path | None = getattr(args, "hook_dll", None)
        if hook_dll is not None:
            hook_image(hook_dll)
        code_version()
        keep_source()
        run: Callable[[argparse.Namespace], int] = args.run
        return run(args)
    except (ValueError, OSError, RuntimeError) as failure:
        print(f"error: {failure}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
