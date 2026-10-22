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
# One CALVIN ABC->D cell for LLaVA-VLA: the official two-process protocol, SHARDS shards on one box.
#
# v3 vs v2: the shard count is no longer tied to the GPU count, and a cell no longer assumes it owns
# the box. SHARDS servers are spread round-robin over NGPU GPUs from PORT_BASE upward, so several cells
# can share one box (each on its own port block) and one cell can put several shards on one GPU. The
# CALVIN evaluator is a PyBullet process that keeps one GPU about 30% busy, so 2-4 shards per GPU is
# where the box actually fills up. The arm proof now reads only this cell's own server pids, which is
# what makes concurrent cells on one box safe.
#
# Protocol is calvin_test/evaluate_policy_multiserver.sh verbatim except for the shard count and the
# AWARE switch: the client evaluates the first 500 fixed sequences of calvin_test/question.json (NUM_SEQUENCES=500,
# the same 500-rollout protocol the CoRL version reported), five chained subtasks each,
# EP_LEN=72, action chunk 5, greedy decoding. The metric is the average number of subtasks completed.
#
# Two environments, both on the shared disk so a recycled container loses nothing:
#   $AWARE_CONDA/envs/llava-vla  - the flask model server (one per GPU)
#   $AWARE_CONDA/envs/calvin     - the PyBullet evaluator (one client per server)
#
#   ARM=base  bash calvin_cell.sh            # released policy, hooks not installed
#   ARM=aware ALPHA=0.05 bash calvin_cell.sh # the rule on, everything else identical
set -u
ARM=${ARM:?arm required: base|aware}
ALPHA=${ALPHA:-}
NGPU=${NGPU:-8}
SHARDS=${SHARDS:-$NGPU}
# NSEQ is how many of the 1000 fixed sequences of question.json the client evaluates. The released
# script hard-codes 500; the CALVIN ABC->D protocol and the LLaVA-VLA paper both use all 1000, which is
# what the literature rows of the table are measured on, so table rows are run at 1000 and the strength
# sweep at 500.
NSEQ=${NSEQ:-500}
# SHARD_FROM/SHARD_TO let several boxes share one cell: each box runs the shard indices in its own slice
# of [0, SHARDS), all writing into the same cell directory on the shared disk. The box that runs the last
# slice is the one that aggregates, so pass AGGREGATE=1 to exactly one of them (default: the box that
# owns shard 0). Servers are numbered by shard index, so the port blocks never collide across boxes.
SHARD_FROM=${SHARD_FROM:-0}
SHARD_TO=${SHARD_TO:-$((SHARDS - 1))}
AGGREGATE=${AGGREGATE:-auto}
PORT_BASE=${PORT_BASE:-9097}
# evaluate_policy_multiserver.py splits the 500 sequences with chunk = ceil(500 / SHARDS) and then takes
# chunk_idx of the resulting list. For some counts that list is shorter than SHARDS and the last shards
# would index past its end, so refuse those counts instead of silently dropping sequences.
_chunk=$(( (NSEQ + SHARDS - 1) / SHARDS ))
_n=$(( (NSEQ + _chunk - 1) / _chunk ))
[ "$_n" = "$SHARDS" ] || { echo "[calvin] SHARDS=$SHARDS splits $NSEQ into $_n chunks of $_chunk; pick another count"; exit 2; }
W=$AWARE_WORK
R=$AWARE_VLA_ROOT/LLaVA-VLA
# MODEL=05b is the CoRL-era backbone (LLaVA-OneVision-Qwen2-0.5B); MODEL=7b is the LLaVA-1.5-7B one.
# They use different servers because OneVision expands the image token differently.
# The 0.5b checkpoint is reached through a named symlink, not the HF snapshot path: llava/builder.py
# dispatches on the directory NAME (it must contain llava and qwen) and a snapshot dir is a commit hash.
MODEL=${MODEL:-05b}
case "$MODEL" in
  05b) SRV=$R/llava/serve/flask_server_qwen.py
       CKPT=$LLAVA_CKPT ;;
  7b)  SRV=$R/llava/serve/flask_server.py
       CKPT=$(ls -d $HF_HOME/hub/models--chenpyyy--LLaVA-VLA/snapshots/*/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2epoch | head -1) ;;
  *) echo "[calvin] unknown MODEL=$MODEL"; exit 2 ;;
esac
STAT=$CALVIN_DATA/validation/statistics.yaml
DATA=$CALVIN_DATA
PY_SRV=$LLAVA_PY
PY_EVAL=$CALVIN_PY

case "$ARM" in
  base)  [ -n "$ALPHA" ] && { echo "[calvin] ARM=base takes no ALPHA"; exit 2; }; TAG=calvin_abcd_${MODEL}_base ;;
  aware) [ -n "$ALPHA" ] || { echo "[calvin] ARM=aware needs ALPHA"; exit 2; }
         TAG=calvin_abcd_${MODEL}_aware_a$(echo "$ALPHA" | tr -d .) ;;
  *) echo "[calvin] unknown ARM=$ARM"; exit 2 ;;
esac
case "$NSEQ" in
  500) ;;
  *) TAG=${TAG}_n${NSEQ} ;;
esac
OD=$W/calvin_cells/$TAG
[ -f "$OD/DONE" ] && { echo "[calvin] $TAG already DONE"; exit 0; }
[ "$SHARD_FROM" = 0 ] && rm -rf "$OD"
mkdir -p "$OD/logs" "$OD/eval"
[ -d "$CKPT" ] || { echo "[calvin] checkpoint missing"; exit 3; }
[ -f "$STAT" ] || { echo "[calvin] statistics.yaml missing"; exit 3; }
[ -f "$DATA/validation/.hydra/merged_config.yaml" ] || { echo "[calvin] validation config missing"; exit 3; }

echo "[calvin] $TAG start $(date '+%F %T') host=$(hostname) model=$MODEL arm=$ARM alpha=${ALPHA:-<off>} shards=$SHARDS ports=$PORT_BASE+ nseq=$NSEQ"

# ---- model servers, one per GPU -------------------------------------------------------------
AENV=""
[ "$ARM" = aware ] && AENV="AWARE_ALPHA=$ALPHA AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction AWARE_ACT_POS=last"
spids=()
for s in $(seq $SHARD_FROM $SHARD_TO); do
  port=$((PORT_BASE + s - SHARD_FROM)); g=$(((s - SHARD_FROM) % NGPU))
  env $AENV CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$R \
      "$PY_SRV" "$SRV" \
      --model-path "$CKPT" --action_stat "$STAT" --port $port \
      > "$OD/logs/server_$s.log" 2>&1 &
  spids+=($!)
done
echo "[calvin] ${#spids[@]} servers launching, waiting for them to answer"

# ---- arm proof: read the switch back out of the servers' own /proc environment ---------------
sleep 90
n=0; ok=0
for pid in "${spids[@]}"; do
  d=/proc/$pid
  [ -r "$d/cmdline" ] || continue
  n=$((n + 1))
  e=$(tr '\0' '\n' < "$d/environ" 2>/dev/null) || e=""
  hit=0
  if [ "$ARM" = base ]; then
    case "$e" in *AWARE_ALPHA=*) ;; *) hit=1 ;; esac
  else
    case "$e" in *"AWARE_ALPHA=$ALPHA"*) hit=1 ;; esac
    case "$e" in *AWARE_TRIGGER=soft*) ;; *) hit=0 ;; esac
    case "$e" in *AWARE_SOFT_DEPTH=ramp*) ;; *) hit=0 ;; esac
    case "$e" in *AWARE_SELECT=instruction*) ;; *) hit=0 ;; esac
  fi
  ok=$((ok + hit))
done
{ echo "model=$MODEL"; echo "arm=$ARM"; echo "alpha=${ALPHA:-}"; echo "shards=$SHARDS"; echo "port_base=$PORT_BASE"
  echo "servers=$n"; echo "with_expected=$ok"
  _mine=$((SHARD_TO - SHARD_FROM + 1))
  if [ "$n" = "$_mine" ] && [ "$ok" = "$_mine" ]; then echo "VERDICT=OK"; else echo "VERDICT=MISMATCH"; fi
} > "$OD/ARM_PROOF.$SHARD_FROM"
cat "$OD/ARM_PROOF.$SHARD_FROM"
grep -q VERDICT=OK "$OD/ARM_PROOF.$SHARD_FROM" || { echo "[calvin] ABORT: arm proof failed"
  for p in "${spids[@]}"; do kill "$p" 2>/dev/null; done; exit 4; }

# wait until every server answers on its port
for s in $(seq $SHARD_FROM $SHARD_TO); do
  port=$((PORT_BASE + s - SHARD_FROM))
  for i in $(seq 1 120); do
    (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null && { exec 3<&-; break; }
    sleep 5
  done
done
echo "[calvin] servers up $(date '+%F %T')"

# ---- evaluation clients, one per server ------------------------------------------------------
cpids=()
for s in $(seq $SHARD_FROM $SHARD_TO); do
  port=$((PORT_BASE + s - SHARD_FROM)); g=$(((s - SHARD_FROM) % NGPU))
  ( cd "$R" && EGL_VISIBLE_DEVICES=$g CUDA_VISIBLE_DEVICES=$g CALVIN_NUM_SEQUENCES=$NSEQ \
    "$PY_EVAL" "$R/calvin_test/evaluate_policy_multiserver.py" \
      --dataset_path "$DATA" \
      --question_file "$R/calvin_test/question.json" \
      --eval_log_dir "$OD/eval" \
      --num_chunks $SHARDS --chunk_idx $s \
      --port $port --server_ip 127.0.0.1 \
      --save_dir "$OD/video" --action_chunk 5 \
      --conf_dir "$CALVIN_CONF" \
      --custom_model ) > "$OD/logs/eval_$g.log" 2>&1 &
  cpids+=($!)
  sleep 2
done
echo "[calvin] ${#cpids[@]} evaluators started $(date '+%F %T')"
for p in "${cpids[@]}"; do wait "$p"; done
echo "[calvin] evaluation finished $(date '+%F %T')"
for p in "${spids[@]}"; do kill "$p" 2>/dev/null; done

# ---- the rule must be proven in every server log ---------------------------------------------
if [ "$ARM" = aware ]; then
  _mine_logs=""
  for s in $(seq $SHARD_FROM $SHARD_TO); do _mine_logs="$_mine_logs $OD/logs/server_$s.log"; done
  AI=$(grep -l "\[AWARE-llavavla\] installed alpha=$ALPHA" $_mine_logs 2>/dev/null | wc -l)
  AP=$(grep -l "\[AWARE-llavavla\] layout proof OK" $_mine_logs 2>/dev/null | wc -l)
  AF=$(grep -l "\[AWARE\] ACTIVE" $_mine_logs 2>/dev/null | wc -l)
  { echo "aware_installed_logs=$AI/$_mine"; echo "aware_layout_proof_logs=$AP/$_mine"; echo "aware_fired_logs=$AF/$_mine"; } >> "$OD/ARM_PROOF.$SHARD_FROM"
  echo "[calvin] aware check: installed=$AI/$_mine layout=$AP/$_mine fired=$AF/$_mine"
  [ "$AI" = "$_mine" ] && [ "$AP" = "$_mine" ] && [ "$AF" = "$_mine" ] || { echo "[calvin] ABORT: AWARE not proven"; exit 7; }
else
  _mine_logs=""; for s in $(seq $SHARD_FROM $SHARD_TO); do _mine_logs="$_mine_logs $OD/logs/server_$s.log"; done
  bad=$(grep -l "AWARE" $_mine_logs 2>/dev/null | wc -l)
  echo "aware_lines_in_base_logs=$bad" >> "$OD/ARM_PROOF"
  [ "$bad" = 0 ] || { echo "[calvin] ABORT: base cell shows AWARE lines"; exit 7; }
fi

# ---- wait for the other boxes' slices, then aggregate (one box only) ---------------------------
if [ "$AGGREGATE" = auto ]; then [ "$SHARD_FROM" = 0 ] && AGGREGATE=1 || AGGREGATE=0; fi
if [ "$AGGREGATE" != 1 ]; then
  echo "[calvin] slice $SHARD_FROM-$SHARD_TO done; another box aggregates"; touch "$OD/SLICE_DONE.$SHARD_FROM"; exit 0
fi
touch "$OD/SLICE_DONE.$SHARD_FROM"
for i in $(seq 1 720); do
  have=$(ls "$OD"/SLICE_DONE.* 2>/dev/null | wc -l)
  files=$(ls "$OD"/eval/results/*.jsonl 2>/dev/null | wc -l)
  [ "$files" -ge "$SHARDS" ] && break
  sleep 30
done

# ---- aggregate -------------------------------------------------------------------------------
MODEL="$MODEL" NSEQ="$NSEQ" "$PY_EVAL" - "$OD" "$TAG" "$ARM" "${ALPHA:-}" <<'PY'
import glob, json, os, sys
od, tag, arm, alpha = sys.argv[1:5]
rows = []
for f in sorted(glob.glob(os.path.join(od, "eval", "**", "results", "*.jsonl"), recursive=True)):
    vals = [json.loads(l) for l in open(f) if l.strip()]
    # the writer appends the chunk mean as a final element; drop it and keep per-sequence counts
    if vals and isinstance(vals[-1], float):
        vals = vals[:-1]
    rows.extend(int(v) for v in vals)
n = len(rows)
avg = sum(rows) / n if n else 0.0
chain = [sum(1 for v in rows if v > k) / n for k in range(5)] if n else []
res = {"tag": tag, "model": os.environ.get("MODEL", ""), "arm": arm, "alpha": alpha, "sequences": n,
       "avg_success_length": round(avg, 4), "n_requested": int(os.environ.get("NSEQ", 500)),
       "chain_sr": [round(100 * c, 2) for c in chain]}
json.dump(res, open(os.path.join(od, "RESULT.json"), "w"), indent=1)
with open(os.path.join(od, "sequences.txt"), "w") as f:
    f.write("\n".join(str(v) for v in rows) + "\n")
print(json.dumps(res, indent=1))
sys.exit(0 if n == int(os.environ.get("NSEQ", 500)) else 5)
PY
rc=$?
[ "$rc" = 0 ] || { echo "[calvin] $TAG BAD: expected $NSEQ sequences"; exit 5; }
touch "$OD/DONE"
echo "[calvin] $TAG done $(date '+%F %T')"
