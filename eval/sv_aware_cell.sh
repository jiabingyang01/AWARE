#!/usr/bin/env bash
# --- AWARE: host paths, override in the environment ------------------------------------------------
: "${AWARE_VLA_ROOT:=${AWARE_VLA_ROOT:-./third_party}}"
: "${AWARE_WORK:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${AWARE_CONDA:=/opt/miniconda3}"
: "${CALVIN_PY:=$AWARE_CONDA/envs/calvin/bin/python}"
: "${LLAVA_PY:=$AWARE_CONDA/envs/llava-vla/bin/python}"
: "${CALVIN_DATA:=$AWARE_VLA_ROOT/../datasets/calvin}"
: "${CALVIN_CONF:=$AWARE_VLA_ROOT/calvin/calvin_models/conf}"
: "${LLAVA_CKPT:?set LLAVA_CKPT to the LLaVA-VLA checkpoint directory}"
# ---------------------------------------------------------------------------------------------------
# One SpatialVLA + AWARE cell over the four SIMPLER-WidowX (Bridge) tasks: sv_cell.sh (the released
# protocol, copied verbatim below) with the AWARE switch added and everything else identical to
# ARM=base: no crop, released temporal ensemble, greedy decoding, deterministic env.
# ARM=aware ALPHA=0.05 bash sv_aware_cell.sh   (SEED is inert here; kept in the tag for bookkeeping)
#
# The protocol is scripts/bridge.sh verbatim, and that script is NOT uniform across its four tasks:
# the eggplant task switches scene, robot, overlay, initial pose and step budget. Running it with the
# first three tasks' setup puts the policy in front of the wrong camera on the wrong table, which
# shows up as "moved the object, never grasped it". Hence the per-task table below rather than one
# shared block. Everything in it is copied from bridge.sh lines 10-51.
#
# Twenty-four episodes per task is not a sample size we chose: put_on_in_scene.py builds twelve xy
# configs and two quat configs and indexes them as episode_id % 24, so 0..23 enumerates every
# distinct initial state exactly once and episode 24 repeats episode 0. Ninety-six per arm is the
# ceiling of this protocol; the only honest way to buy power is to repeat the stochastic arm.
#
# Each task's range is split in half across two cards (the evaluator iterates range(start, end), so
# 0..11 and 12..23 are disjoint and their union is the released 24), which fills all eight GPUs and
# changes no episode and no initial state.
#
# ARM: base (switch off, the released deployment) | centre (crop pinned) | random (the intervention).
# SEED only matters for random; base and centre decode greedily on a deterministic env.
#
# SAPIEN: SIGTERM only, never SIGKILL -- a killed renderer wedges the NVIDIA driver for the machine.
set -u
ARM=${ARM:?arm required: aware}
ALPHA=${ALPHA:?AWARE alpha required, e.g. 0.05}
SEED=${SEED:-7}
EPS=${EPS:-24}
B=${B:-$AWARE_WORK}
V=$AWARE_VLA_ROOT/SimplerEnv-SpatialVLA
PY=/opt/spatialvla_venv/bin/python
CKPT=IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge
SCALE=${SPATIALVLA_CROP_SCALE:-0.9}
DATA=$V/ManiSkill2_real2sim/data

case "$ARM" in
  aware) WANT="" ;;
  *) echo "[sv] this launcher runs ARM=aware only (base/centre/random: sv_cell.sh)"; exit 2 ;;
esac
ANODOT=$(echo "$ALPHA" | tr -d .)
# TAGSUF keeps an ablation's cells in their own directories so they can never be mistaken for the
# main arms. NOENS=1 turns the temporal action ensemble off inside the policy; it is the mechanism
# ablation SPATIALVLA_PREREG.md rule 4 permits only after the decorrelation contrast came out null.
TAGSUF=${TAGSUF:-}
NOENS=${NOENS:-}
TAG=sv_bridge_aware_a${ANODOT}_s${SEED}${TAGSUF}
OD=$B/sv_cells/$TAG
if [ -f "$OD/DONE" ]; then echo "[sv] $TAG already DONE"; exit 0; fi
rm -rf "$OD"; mkdir -p "$OD"

# name|env|scene|robot|overlay|init_x|init_y|max_steps   -- straight out of scripts/bridge.sh
TASKS=(
  "PutCarrotOnPlate|PutCarrotOnPlateInScene-v0|bridge_table_1_v1|widowx|bridge_real_eval_1.png|0.147|0.028|60"
  "StackGreenCubeOnYellowCube|StackGreenCubeOnYellowCubeBakedTexInScene-v0|bridge_table_1_v1|widowx|bridge_real_eval_1.png|0.147|0.028|60"
  "PutSpoonOnTableCloth|PutSpoonOnTableClothInScene-v0|bridge_table_1_v1|widowx|bridge_real_eval_1.png|0.147|0.028|60"
  "PutEggplantInBasket|PutEggplantInBasketScene-v0|bridge_table_1_v2|widowx_sink_camera_setup|bridge_sink.png|0.127|0.06|120"
)
for t in "${TASKS[@]}"; do
  IFS='|' read -r _n _e _s _r ov _x _y _m <<< "$t"
  [ -f "$DATA/real_inpainting/$ov" ] || { echo "[sv] overlay missing: $DATA/real_inpainting/$ov"; exit 3; }
done

HALF=$((EPS / 2))
echo "[sv] $TAG start $(date '+%F %T')  AWARE alpha=$ALPHA crop=${WANT:-<off>} eps=$EPS seed=$SEED (8 shards) host=$(hostname)"
pids=()
for i in "${!TASKS[@]}"; do
  IFS='|' read -r NAME ENV SCENE ROBOT OV RX RY MAXS <<< "${TASKS[$i]}"
  for j in 0 1; do
    g=$((i * 2 + j))
    lo=$((j * HALF)); hi=$((lo + HALF)); [ "$j" = 1 ] && hi=$EPS
    if [ -n "$WANT" ]; then
      CROPENV="SPATIALVLA_CROP=$WANT SPATIALVLA_CROP_SCALE=$SCALE SPATIALVLA_CROP_SEED=$((SEED * 100 + g))"
    else
      CROPENV=""
    fi
    [ "$NOENS" = 1 ] && CROPENV="$CROPENV SPATIALVLA_NO_ENSEMBLE=1"
    # The same switch set every other backbone runs under (run_aware_cogact.py, the OFT and pi0
    # runners): soft entropy-weighted blend, depth ramp, instruction-weighted retrieval.
    CROPENV="$CROPENV AWARE_ALPHA=$ALPHA AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction AWARE_ACT_POS=last"
    env $CROPENV CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        "$PY" "$V/simpler_env/main_inference.py" \
        --policy-model spatialvla --ckpt-path "$CKPT" --action-ensemble-temp -0.8 \
        --logging-dir "$OD/task${i}_h${j}" \
        --robot "$ROBOT" --policy-setup widowx_bridge \
        --control-freq 5 --sim-freq 500 --max-episode-steps "$MAXS" \
        --env-name "$ENV" --scene-name "$SCENE" \
        --rgb-overlay-path "$DATA/real_inpainting/$OV" \
        --robot-init-x "$RX" "$RX" 1 --robot-init-y "$RY" "$RY" 1 \
        --obj-variation-mode episode --obj-episode-range "$lo" "$hi" \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        > "$OD/task${i}_h${j}.log" 2>&1 &
    pids+=($!)
    sleep 3
  done
done

# Self-proof: read the arm switch back out of the workers' own /proc environment. A cell whose
# workers do not carry the flag is not the arm it claims to be, whatever the launcher intended.
sleep 60
n=0; ok=0
for d in /proc/[0-9]*; do
  [ -r "$d/cmdline" ] || continue
  c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$c" in *main_inference*) ;; *) continue ;; esac
  case "$c" in *python*) ;; *) continue ;; esac
  n=$((n + 1))
  e=$(tr '\0' '\n' < "$d/environ" 2>/dev/null) || e=""
  hit=0
  case "$e" in *"AWARE_ALPHA=$ALPHA"*) hit=1 ;; esac
  case "$e" in *AWARE_TRIGGER=soft*) ;; *) hit=0 ;; esac
  case "$e" in *AWARE_SOFT_DEPTH=ramp*) ;; *) hit=0 ;; esac
  case "$e" in *AWARE_SELECT=instruction*) ;; *) hit=0 ;; esac
  case "$e" in *SPATIALVLA_CROP=*) hit=0 ;; esac
  if [ "$NOENS" = 1 ]; then
    case "$e" in *SPATIALVLA_NO_ENSEMBLE=1*) ;; *) hit=0 ;; esac
  else
    case "$e" in *SPATIALVLA_NO_ENSEMBLE=*) hit=0 ;; esac
  fi
  ok=$((ok + hit))
done
{ echo "arm=$ARM alpha=$ALPHA want=${WANT:-<off>}"; echo "no_ensemble=${NOENS:-0}"; echo "seed=$SEED"
  echo "workers=$n"; echo "with_expected=$ok"
  if [ "$n" = 8 ] && [ "$ok" = 8 ]; then echo "VERDICT=OK"; else echo "VERDICT=MISMATCH"; fi
} > "$OD/ARM_PROOF"
cat "$OD/ARM_PROOF"
grep -q VERDICT=OK "$OD/ARM_PROOF" || { echo "[sv] ABORT: arm proof failed"
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; exit 4; }

for p in "${pids[@]}"; do wait "$p"; done
echo "[sv] $TAG all tasks done $(date '+%F %T')"

# What the ensemble actually did, read back out of the workers' own stdout. The /proc arm proof
# above shows the flag was passed; this shows it took effect.
if [ "$NOENS" = 1 ]; then EXP=False; else EXP=True; fi
GOT=$(grep -h -o "action_ensemble=[A-Za-z]*" "$OD"/task*_h*.log 2>/dev/null | sort -u | tr '\n' ' ')
NOK=$(grep -h -c "action_ensemble=$EXP " "$OD"/task*_h*.log 2>/dev/null | awk "{s+=\$1} END{print s+0}")
{ echo "ensemble_expected=$EXP"; echo "ensemble_logged=${GOT:-none}"; echo "shards_agreeing=${NOK:-0}/8"; } \
  >> "$OD/ARM_PROOF"
echo "[sv] ensemble check: expected=$EXP logged=${GOT:-none} shards=${NOK:-0}/8"
[ "${NOK:-0}" = 8 ] || { echo "[sv] ABORT: ensemble state not confirmed in all eight shard logs"; exit 6; }
# The rule must have installed with this alpha, proved the prefix layout and actually fired, in every
# shard's own stdout. A cell whose workers carried the flag but never blended is not the arm it claims.
AI=$(grep -l "\[AWARE-spatialvla\] installed alpha=$ALPHA" "$OD"/task*_h*.log 2>/dev/null | wc -l)
AP=$(grep -l "\[AWARE-spatialvla\] layout proof OK" "$OD"/task*_h*.log 2>/dev/null | wc -l)
AF=$(grep -l "\[AWARE\] ACTIVE" "$OD"/task*_h*.log 2>/dev/null | wc -l)
{ echo "aware_installed_logs=$AI/8"; echo "aware_layout_proof_logs=$AP/8"; echo "aware_fired_logs=$AF/8"; } >> "$OD/ARM_PROOF"
echo "[sv] aware check: installed=$AI/8 layout_proof=$AP/8 fired=$AF/8"
[ "$AI" = 8 ] && [ "$AP" = 8 ] && [ "$AF" = 8 ] || { echo "[sv] ABORT: AWARE not proven in all eight shard logs"; exit 7; }

ALPHA="$ALPHA" "$PY" - "$OD" "$TAG" "$ARM" "$SEED" <<'PY'
import glob, os, re, sys
od, tag, arm, seed = sys.argv[1:5]
TASKS = ["PutCarrotOnPlate", "StackGreenCubeOnYellowCube", "PutSpoonOnTableCloth", "PutEggplantInBasket"]
# The evaluator names each video "{success|failure}_obj_episode_{id}" and then appends the episode
# stats, e.g. "..._moved_correct_obj_False_is_src_obj_grasped_False...". Anchoring on "\.mp4$" right
# after the id silently matches nothing.
PAT = re.compile(r"(success|failure)_obj_episode_(\d+)(?:_.*)?\.mp4$")
rows, per = [], []
for i, name in enumerate(TASKS):
    got = {}
    for h in (glob.glob(os.path.join(od, f"task{i}_h0", "**", "*.mp4"), recursive=True)
              + glob.glob(os.path.join(od, f"task{i}_h1", "**", "*.mp4"), recursive=True)):
        m = PAT.match(os.path.basename(h))
        if m:
            got[int(m.group(2))] = int(m.group(1) == "success")
    for ep in sorted(got):
        rows.append((name, ep, got[ep]))
    per.append((name, sum(got.values()), len(got)))
with open(os.path.join(od, "episodes.tsv"), "w") as f:
    f.write("task\tepisode_id\tsuccess\n")
    for t, e, s in rows:
        f.write(f"{t}\t{e}\t{s}\n")
suc = sum(s for _, _, s in rows); tot = len(rows)
with open(os.path.join(od, "RESULT"), "w") as f:
    f.write(f"tag={tag}\narm={arm}\nalpha={os.environ.get('ALPHA','')}\nseed={seed}\n")
    for name, s, n in per:
        f.write(f"{name}={s}/{n}\n")
    f.write(f"success={suc}/{tot}\n")
    f.write(f"arm_proof={[l for l in open(os.path.join(od,'ARM_PROOF')).read().strip().splitlines() if l.startswith('VERDICT=')][0].split('=')[-1]}\n")
print(open(os.path.join(od, "RESULT")).read(), end="")
sys.exit(0 if tot == 96 else 5)
PY
rc=$?
cat "$OD/RESULT" 2>/dev/null
[ "$rc" = 0 ] || { echo "[sv] $TAG BAD: episode count wrong"; exit 5; }
touch "$OD/DONE"
echo "[sv] $TAG done $(date '+%F %T')"
