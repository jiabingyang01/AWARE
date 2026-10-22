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
# One OpenVLA arm of the block-hold sweep (8x NVIDIA A100 GPUs).
#
# The policy is the official greedy decode of the released LIBERO-10 checkpoint, queried every
# control step (--replan_steps 1). The only thing an arm changes is the crop offset's behaviour
# in time, entirely server-side:
#
#   ARM=center   no CROP_MODE            the official centre crop
#   ARM=aware    AWARE_ALPHA=x           the official centre crop plus AWARE, server-side
#   ARM=hold     CROP_MODE=hold CROP_HOLD=k   one offset held k steps, then redrawn
#
# CROP_HOLD=1 is the per-step arm, and it doubles as the port check: it should land near the
# 59% this suite already gave on the other machine, against 49% for centre.
#
# Required env: ARM(center|hold)  [HOLD=k for ARM=hold]  [SEED]  [SUITE]
set -u
ARM=${ARM:?ARM required}
HOLD=${HOLD:-8}
SEED=${SEED:-7}
SUITE=${SUITE:-libero_10}
TRIALS=${TRIALS:-50}
# NPER pairs per GPU. The other machine ran 3 on 128 cores; this one has 90, and 3 pairs would put
# 24 servers at 4 threads each plus 24 clients against them, which is where that machine thrashed
# and the GPUs idled. 2 pairs is 16 servers and 16 clients, about 80 threads, and it leaves memory
# at roughly 45 GB of 80 per card. The 500-episode set is identical either way: workers shard the
# same (task, trial) grid and their union is the whole suite.
NPER=${NPER:-2}
PORT0=${PORT0:-9700}

B=${B:-$AWARE_WORK}
OP=$AWARE_VLA_ROOT/openpi
SRV=$AWARE_VLA_ROOT/openvla/serve/openvla_server.py   # shared: see that dir README
N=$((8 * NPER))

case $SUITE in
  libero_10)      CK=openvla/openvla-7b-finetuned-libero-10;      KEY=libero_10 ;;
  libero_spatial) CK=openvla/openvla-7b-finetuned-libero-spatial; KEY=libero_spatial ;;
  libero_object)  CK=openvla/openvla-7b-finetuned-libero-object;  KEY=libero_object ;;
  libero_goal)    CK=openvla/openvla-7b-finetuned-libero-goal;    KEY=libero_goal ;;
  *) echo "[ovla] unknown SUITE=$SUITE"; exit 2 ;;
esac

AWAREENV=""
if [ "$ARM" = center ]; then
  TAG=ovla_${SUITE#libero_}_center_s${SEED}; CROPENV=""
elif [ "$ARM" = hold ]; then
  TAG=ovla_${SUITE#libero_}_hold${HOLD}_s${SEED}; CROPENV="CROP_MODE=hold CROP_HOLD=$HOLD"
elif [ "$ARM" = aware ]; then
  # The image path is the official one; AWARE changes only what the LLM reads while it decodes
  # the seven action tokens. ALPHA=0 is the control: the hooks install and blend nothing.
  A=${ALPHA:?ALPHA required for ARM=aware}
  TAG=ovla_${SUITE#libero_}_aware$(echo "$A" | tr -d '.')_s${SEED}; CROPENV=""
  AWAREENV="AWARE_ALPHA=$A AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction"
else
  echo "[ovla] unknown ARM=$ARM"; exit 2
fi
OUT=$B/ovla_cells/$TAG
mkdir -p "$OUT/logs" "$OUT/cl"

# Weights must already be local. Every server re-validating repo metadata over the network is a
# way to fail a launch, not a way to help it.
export HF_HOME=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
SNAP=$(ls -d $HF_HUB_CACHE/models--openvla--openvla-7b-finetuned-${KEY//_/-}/snapshots/*/ 2>/dev/null | head -1)
[ -n "${SNAP:-}" ] || { echo "[ovla] ABORT: $CK not in $HF_HUB_CACHE"; exit 3; }
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

if ss -ltn 2>/dev/null | awk -v p0=$PORT0 -v n=$N 'BEGIN{bad=0} {split($4,a,":"); pt=a[length(a)]+0; if (pt>=p0 && pt<p0+n) bad=1} END{exit !bad}'; then
  echo "[ovla] ABORT: ports $PORT0..$((PORT0+N-1)) not all free"; exit 3
fi

# Thread caps: 24 unconstrained server processes each open a pool sized from the core count and
# the box thrashes while the GPUs idle.
TH="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
CTH="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EPTAG=1"

echo "[ovla] $TAG  suite=$SUITE ckpt=$CK seed=$SEED N=$N  start $(date '+%F %T')"
echo "[ovla] crop env: ${CROPENV:-<none, official centre crop>}"
: > "$OUT/server_pids"
for i in $(seq 0 $((N - 1))); do
  env CUDA_VISIBLE_DEVICES=$((i % 8)) PORT=$((PORT0 + i)) PYTHONPATH=$OP/src \
    $TH OVLA_CKPT=$CK OVLA_UNNORM_KEY=$KEY PERTURB_MODE=crop $CROPENV $AWAREENV \
    $OP/.venv/bin/python "$SRV" > "$OUT/logs/server$i.log" 2>&1 < /dev/null &
  echo $! >> "$OUT/server_pids"
  sleep 0.4
done

R=0
for _ in $(seq 1 300); do
  R=$(grep -l 'ready on' "$OUT"/logs/server*.log 2>/dev/null | wc -l); [ "$R" = "$N" ] && break; sleep 5
done
echo "[ovla] servers ready = $R/$N"

# Self-proof from the servers' own environment, the same gate the pi0 cells use. A switch that
# cannot show itself in the log is a cell nobody can audit later.
n_srv=0; n_flag=0
for d in /proc/[0-9]*; do
  [ -r "$d/cmdline" ] || continue
  c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$c" in *openvla_server*) ;; *) continue ;; esac
  case "$c" in *python*) ;; *) continue ;; esac
  n_srv=$((n_srv + 1))
  e=$(tr '\0' '\n' < "$d/environ" 2>/dev/null) || e=""
  if [ "$ARM" = hold ]; then
    case "$e" in *CROP_MODE=hold*) case "$e" in *CROP_HOLD=$HOLD*) n_flag=$((n_flag + 1)) ;; esac ;; esac
  elif [ "$ARM" = aware ]; then
    case "$e" in *AWARE_ALPHA=$A*) case "$e" in *CROP_MODE=*) ;; *) n_flag=$((n_flag + 1)) ;; esac ;; esac
  else
    case "$e" in *CROP_MODE=*) ;; *) n_flag=$((n_flag + 1)) ;; esac
  fi
done
n_inst=$N
if [ "$ARM" = aware ]; then
  n_inst=$(grep -l '\[AWARE-openvla\] installed' "$OUT"/logs/server*.log 2>/dev/null | wc -l)
fi
{ echo "arm=$ARM hold=$HOLD alpha=${ALPHA:-}"; echo "servers_seen=$n_srv"; echo "with_expected_env=$n_flag"
  echo "servers_logging_install=$n_inst"; echo "expected=$N"
  [ "$n_flag" = "$N" ] && [ "$n_inst" = "$N" ] && echo "VERDICT=OK" || echo "VERDICT=MISMATCH"; } > "$OUT/ARM_PROOF"
cat "$OUT/ARM_PROOF"

if [ "$R" != "$N" ] || [ "$n_flag" != "$N" ] || [ "$n_inst" != "$N" ]; then
  echo "[ovla] ABORT before clients: ready=$R/$N env=$n_flag/$N installed=$n_inst/$N"
  while read -r pid; do kill "$pid" 2>/dev/null; done < "$OUT/server_pids"
  exit 4
fi

# --resize_size 256 hands the raw rendered frame over; the official 256->224 resize and the
# 0.9-area crop happen server-side, so downsampling client-side would put the policy out of
# distribution.
EGL_ICD_JSON=$OP/examples/libero/10_nvidia.json
cl_pids=""   # must exist before first append: the script runs under set -u
for i in $(seq 0 $((N - 1))); do
  ( cd "$OP" && exec env $CTH \
      PYTHONPATH="$OP/third_party/libero" \
      MUJOCO_EGL_DEVICE_ID=$((i % 8)) __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_JSON" \
      "$OP/examples/libero/.venv/bin/python" examples/libero/main_parallel.py \
      --host 127.0.0.1 --port $((PORT0 + i)) --task_suite_name "$SUITE" \
      --num_trials_per_task "$TRIALS" --worker_id "$i" --num_workers "$N" --seed "$SEED" \
      --replan_steps 1 --resize_size 256 \
      --local_log_dir "$OUT/cl" --run_id "$TAG" > "$OUT/cl/w$i.log" 2>&1 < /dev/null ) &
  cl_pids="$cl_pids $!"
  sleep 0.4
done
echo "[ovla] $N clients launched"
# Bare `wait` also waits on the 16 resident servers, which never exit -- the cell then hangs
# after every episode is already on disk (hit once on ovla_10_center_s7). Wait on clients only.
for p in $cl_pids; do wait "$p"; done
echo "[ovla] clients finished $(date '+%F %T')"

while read -r pid; do kill "$pid" 2>/dev/null; done < "$OUT/server_pids"
sleep 10
while read -r pid; do kill -9 "$pid" 2>/dev/null; done < "$OUT/server_pids"

NEP=$(cat "$OUT"/cl/*worker*.txt 2>/dev/null | grep -c 'success=')
NSU=$(cat "$OUT"/cl/*worker*.txt 2>/dev/null | grep -c 'success=True')
WANT=$((TRIALS * 10))
{ echo "tag=$TAG"; echo "arm=$ARM"; echo "hold=$HOLD"; echo "alpha=${ALPHA:-}"; echo "seed=$SEED"
  echo "success=$NSU/$NEP"; echo "arm_proof=$(grep -h VERDICT "$OUT/ARM_PROOF" | cut -d= -f2)"
  echo "finished=$(date '+%F %T')"; } > "$OUT/RESULT"
echo "[ovla] $TAG = $NSU/$NEP  (want $WANT)"
[ "$NEP" = "$WANT" ] || { echo "[ovla] $TAG BAD: $NEP episodes, want $WANT"; exit 5; }
# A count of 500 is not proof that 500 episodes ran. When the server raises on every inference the
# clients still write one record per episode, all of them failures, and the cell finishes in a
# minute with a 0/500 that passes the episode count. The released arm scores about half this suite,
# so zero successes is a broken cell, never a measurement.
[ "$NSU" -gt 0 ] || { echo "[ovla] $TAG BAD: 0 successes -- the policy errored, see logs/"; exit 6; }
touch "$OUT/DONE"
