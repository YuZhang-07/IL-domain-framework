#!/usr/bin/env bash
# Run the supplied AMBER stages; topology generation is a separate local step.
# No input keyword values are changed by this wrapper. See md/README.md.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'HELP'
Usage: bash md/scripts/run_md.sh [options]

  --work-dir DIR     Directory with prepared cmp.prmtop and cmp.inpcrd.
                    Default: the current working directory.
  --input-dir DIR    Directory containing the seven AMBER .in files.
                    Default: md/input_templates/amber beside this script.
  --topology FILE    Override the prepared topology path.
  --coordinates FILE
                    Override the initial coordinate path.
  --output-dir DIR   New or empty output directory.
                    Default: WORK_DIR/md_output.
  --pmemd PROGRAM   Executable name or path (default: PMEMD_CUDA or pmemd.cuda).
  --gpu VALUE       Explicit CUDA_VISIBLE_DEVICES override. Otherwise preserve
                    an existing setting, including an empty one; if unset use 0.
  --dry-run         Print the 19 commands without executing MD or creating outputs.
                    The input templates/topology/coordinates must still exist;
                    an installed AMBER executable is not required for this mode.
  -h, --help        Show help.

All explicitly supplied relative paths are relative to the calling directory.
Output names and stage order match the supplied run.sh. This is a fresh-run
wrapper, not a resume tool. Packmol, tleap, parameters and topology validation
are not performed here. Trajectory writing during a local run is unchanged.
HELP
}
need_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
}
absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$CALL_DIR" "$1" ;;
    esac
}
readable_file() {
    [[ -f "$1" && -r "$1" && -s "$1" ]] || fail "Missing, empty or unreadable file: $1"
}

CALL_DIR=$(pwd -P)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORK_DIR="$CALL_DIR"
INPUT_DIR="$SCRIPT_DIR/../input_templates/amber"
TOPOLOGY=''
COORDINATES=''
OUTPUT_DIR=''
PMEMD="${PMEMD_CUDA:-pmemd.cuda}"
GPU=''
GPU_SET=0
DRY_RUN=0

while (( $# )); do
    case "$1" in
        --work-dir)    need_value "$@"; WORK_DIR=$2; shift 2 ;;
        --input-dir)   need_value "$@"; INPUT_DIR=$2; shift 2 ;;
        --topology)    need_value "$@"; TOPOLOGY=$2; shift 2 ;;
        --coordinates) need_value "$@"; COORDINATES=$2; shift 2 ;;
        --output-dir)  need_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
        --pmemd)       need_value "$@"; PMEMD=$2; shift 2 ;;
        --gpu)         need_value "$@"; GPU=$2; GPU_SET=1; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             fail "Unknown argument: $1 (use --help)" ;;
    esac
done

WORK_DIR=$(absolute_path "$WORK_DIR")
INPUT_DIR=$(absolute_path "$INPUT_DIR")
[[ -d "$WORK_DIR" ]] || fail "Working directory does not exist: $WORK_DIR"
[[ -d "$INPUT_DIR" ]] || fail "Input directory does not exist: $INPUT_DIR"
WORK_DIR=$(cd -- "$WORK_DIR" && pwd -P)
INPUT_DIR=$(cd -- "$INPUT_DIR" && pwd -P)
TOPOLOGY=$(absolute_path "${TOPOLOGY:-$WORK_DIR/cmp.prmtop}")
COORDINATES=$(absolute_path "${COORDINATES:-$WORK_DIR/cmp.inpcrd}")
OUTPUT_DIR=$(absolute_path "${OUTPUT_DIR:-$WORK_DIR/md_output}")
readable_file "$TOPOLOGY"
readable_file "$COORDINATES"
for name in min1 min2 min3 heat eq1 eq2 eq3; do
    readable_file "$INPUT_DIR/$name.in"
done

# Do not let AMBER's -O overwrite files from an earlier run.
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    [[ -d "$OUTPUT_DIR" ]] || fail "Output path is not a directory: $OUTPUT_DIR"
    shopt -s nullglob dotglob
    entries=("$OUTPUT_DIR"/*)
    shopt -u nullglob dotglob
    (( ${#entries[@]} == 0 )) || fail "Output directory is not empty; choose a new directory."
fi

# Respect the scheduler/caller's CUDA assignment unless explicitly overridden.
if (( GPU_SET )); then
    export CUDA_VISIBLE_DEVICES="$GPU"
elif [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
if [[ "$PMEMD" == */* ]]; then
    PMEMD=$(absolute_path "$PMEMD")
else
    if resolved=$(command -v "$PMEMD"); then
        PMEMD=$(absolute_path "$resolved")
    elif (( ! DRY_RUN )); then
        fail "AMBER executable not found. Load your local environment or set --pmemd."
    fi
fi
if (( ! DRY_RUN )); then
    [[ -f "$PMEMD" && -x "$PMEMD" ]] || fail "AMBER executable is not runnable: $PMEMD"
    mkdir -p -- "$OUTPUT_DIR"
    cd -- "$OUTPUT_DIR"
fi

# This warning describes an unresolved source setting; do not silently enable
# a temperature schedule or claim that the original heating was validated.
printf '%s\n' 'NOTE: See md/README.md for the supplied heat.in TEMP0/nmropt warning.' >&2
if (( DRY_RUN )); then
    printf '# DRY RUN: no MD executed; commands would run in %q\n' "$OUTPUT_DIR"
fi

run_stage() {
    local input_name=$1 previous=$2 stem=$3 write_trajectory=$4
    local -a cmd=("$PMEMD" -O -i "$INPUT_DIR/$input_name" -p "$TOPOLOGY"
                  -c "$previous" -ref "$previous" -r "$stem.rst" -o "$stem.out")
    if (( write_trajectory )); then
        cmd+=(-x "$stem.mdcrd")
    fi
    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    readable_file "$previous"
    printf 'Running %s using %s\n' "$stem" "$input_name"
    "${cmd[@]}"
    readable_file "$stem.rst"
}

# Preserve all 19 calls, their references, and the original output stems.
run_stage min1.in "$COORDINATES" cmp_min1 0
run_stage min2.in cmp_min1.rst cmp_min2 0
run_stage min3.in cmp_min2.rst cmp_min3 0
run_stage heat.in cmp_min3.rst cmp_heat 1
run_stage eq3.in cmp_heat.rst cmp_eq13 1
run_stage eq3.in cmp_eq13.rst cmp_eq14 1
run_stage eq3.in cmp_eq14.rst cmp_eq15 1
run_stage eq3.in cmp_eq15.rst cmp_eq16 1
run_stage eq1.in cmp_eq16.rst cmp_eq0 1
for (( segment=1; segment<=10; segment++ )); do
    previous=$((segment - 1))
    run_stage eq2.in "cmp_eq${previous}.rst" "cmp_eq${segment}" 1
done

if (( ! DRY_RUN )); then
    printf 'All commands finished; outputs are in the selected output directory.\n'
fi
