# MD input templates and run script

This is a **selected publication subset**, not a self-contained reconstruction of
all simulated systems. It contains the 11 supplied MD-related files, organized
for publication, plus this documentation and local Git rules. No trajectories,
restart datasets, complete protein systems, or additional force-field files are
included. No files are downloaded or uploaded by the run script.

[中文说明](README_zh-CN.md)

## Layout

```text
md/
├── README.md
├── README_zh-CN.md
├── .gitignore
├── .gitattributes
├── scripts/
│   └── run_md.sh
├── input_templates/
│   ├── amber/
│   │   ├── min1.in
│   │   ├── min2.in
│   │   ├── min3.in
│   │   ├── heat.in
│   │   ├── eq1.in
│   │   ├── eq2.in
│   │   └── eq3.in
│   ├── packmol/
│   │   └── packmol_example.inp
│   └── tleap/
│       └── tleap_example.in
└── structures/
    └── Bmim.pdb
```

Place `md/` directly in the existing repository root, beside `scripts/`,
`configs/` and `docs/`. Do not duplicate it under Task 1 or Task 2. The `.in` and
`.inp` files already express the simulation settings; no JSON/YAML conversion
or additional configuration framework is needed.

## What was changed

The seven AMBER inputs retain **every supplied keyword and value**, including
restraint masks, step counts, random-seed settings, temperature/pressure controls,
cutoff and output frequencies. Inline comments and trailing whitespace were
removed. `eq2.in` received a descriptive first-line title; the final
`&wt TYPE='END'` block in `heat.in` received its missing `/`; the two restraint
assignments in each of `min1.in` and `min2.in` received trailing commas.
These are documented text/syntax edits, not a claim that the original inputs
were valid or that the resulting MD trajectories have been reproduced.

`packmol_example.inp` replaces only the example-specific protein filename
`214_3.30.200.20_2.60.40.10.pdb` with `protein.pdb`, while retaining the fixed
placement, packing tolerance, molecule counts and box values. `tleap_example.in`
retains all original executable statements in their original order. Both files
use local working-directory filenames. `Bmim.pdb` is byte-for-byte unchanged.

`run_md.sh` adds command-line paths, shell quoting, error stopping, and a new/empty
output-directory requirement. It replaces the hard-coded GPU override with an
optional `--gpu`: a pre-existing `CUDA_VISIBLE_DEVICES` setting is preserved
unless explicitly overridden; the original default of `0` is used only when the
variable is unset. It preserves the 19 AMBER calls, all output stems and restart
references. It does not run Packmol/tleap or change AMBER input keywords.

## Scope and retained warnings

**Heating:** the supplied `heat.in` contains a `TEMP0` weight block but does not
explicitly set `nmropt` in `&cntrl`. The published copy deliberately does not add
`nmropt`, `tempi` or `temp0`. Thus this package does **not** establish that the
stated 10-to-300 temperature schedule was active in the original runs. The Amber
official [heating example](https://ambermd.org/tutorials/basic/tutorial5/index.php)
explicitly uses `nmropt=1` with `&wt TYPE='TEMP0'`; the source omission is recorded,
not silently converted into a different protocol. The runner prints a reminder.

**Example-specific settings:** Packmol uses `inside box 0 0 0 65 65 65`; tleap uses
`set com box {67.0, 67.0, 67.0}`. Both original values remain; they were not made
equal. The example also retains the `:1-25` and `:1-25@CA,C,N` restraint masks.
Neither the boxes nor the masks are universal settings for every protein.
The Packmol file contains 5000 water copies and 100 `Bmim.pdb` copies; the original
tleap ion-addition commands remain unchanged. No missing ion or parameter file
was inferred or generated.

**Local dependencies, not additional files in this release:** Packmol needs
`protein.pdb` and `water.pdb` as well as the supplied `Bmim.pdb`. Tleap needs
`Bmim.frcmod`, `Bmim.mol2` and the packed `BMIMS_pro.pdb`. The run script starts
from prepared `cmp.prmtop` and `cmp.inpcrd`. The PDB alone is not a force-field
parameterization. AMBER, Packmol, their installed force-field resources, and
hardware/software versions are not supplied or inferred from these files.

These limitations do not prevent publishing the selected input files, but the
subset must not be described as a complete, independently runnable MD dataset
or a reproduction of the paper's trajectory-derived results.

## Local use with already available dependencies

The following is for a local installation with the missing dependencies already
available; publication alone does not require rerunning MD. Keep local run data
in `md/work/system_example/` or outside the repository.

For model preparation, copy the Packmol/tleap examples and `structures/Bmim.pdb`
into the same working directory as the local protein/water/parameter files, using
`packmol.inp`, `tleap.in` and `Bmim.pdb` as the local names. Invoke the tools from
that working directory:

```bash
packmol < packmol.inp
tleap -f tleap.in
```

After valid topology/coordinate files exist, run from the repository root:

```bash
bash md/scripts/run_md.sh \
  --work-dir md/work/system_example \
  --output-dir md/work/system_example/md_output
```

The default executable is `pmemd.cuda` (or the `PMEMD_CUDA` environment variable).
Use `--pmemd` for an explicit executable. `--help` documents `--input-dir`,
`--topology`, `--coordinates`, `--gpu` and `--dry-run`. All explicitly supplied
relative paths are resolved from the calling directory. Default topology and
coordinates are under `--work-dir`; the default output is `WORK_DIR/md_output`.
`--dry-run` requires the input files but does not require AMBER or create outputs.
This is a fresh-run wrapper, not a continuation/resume implementation.

The original sequence is retained:

```text
min1 -> min2 -> min3 -> heat
     -> eq3 (cmp_eq13 ... cmp_eq16, four calls)
     -> eq1 (cmp_eq0)
     -> eq2 (cmp_eq1 ... cmp_eq10, ten sequential calls)
```

The ten `eq2.in` segments use the supplied `nstlim=5000000, dt=0.002`.
Trajectory-writing arguments and frequencies remain enabled during local runs;
only publication of the resulting trajectory files is excluded.

##snapshots
The MD/snapshots/ directory contains selected PDB structural snapshots extracted from the molecular dynamics simulations under water and BmimCl conditions. These structures are provided for visualization and structural inspection and do not represent the complete MD trajectories.

## Publication and checks

The `md/.gitignore` allowlist includes only the 15 release files. It does not
replace the repository-root Git rules, untrack already committed files, or filter
manual browser uploads. Upload the clean release folder, not a populated local
MD working directory. Keep both hidden Git files inside `md/`.

Verification performed: input keyword/value comparison; unchanged PDB bytes;
Packmol/tleap command comparison (except the documented protein filename);
`bash -n`; and an original-versus-refactored **mock-executable** trace comparison
of the 19 calls. Empty-output protection, error stopping and GPU-setting handling
were exercised with mock files. AMBER, tleap and Packmol were **not installed or
executed** during these checks; no MD calculation or physical validation occurred.

The selected originals and released code/inputs were checked for common private
key blocks, credential/token patterns, hard-coded personal paths and obvious
internal host identifiers; none of the checked patterns were found. This is a
scoped review, not a guarantee about unprovided files or Git history. Local
console output, `--dry-run` commands, engine logs and exception messages may
contain local paths and are not automatically sanitized for publication.
