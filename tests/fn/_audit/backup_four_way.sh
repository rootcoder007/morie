#!/usr/bin/env bash
# Four-way backup of the tests/fn audit artifacts. Run at the end of every cluster.
#
# VSR (Mac)      canonical working copy -- already there, nothing to do
# zeus           bare git mirror + GRAD_STUDIES rsync
# L14 internal   ~/backups/<stamp>/   staging; always present
# WD_BLACK       fourth copy; located via findmnt, NOT a hardcoded path
#
# The WD is desktop-mounted by udisks at /run/media/<user>/WD_BLACK, not at
# /mnt/wdblack. On 2026-07-26 I probed /mnt/wdblack, found a plain directory on
# the internal disk, and concluded the drive was absent -- an rsync there would
# have written to /home and reported success. Hence: locate, never assume.
set -uo pipefail

ROOT=/Volumes/VSR/rootcoderfiles
STAMP=${1:-$(date -u +%Y-%m-%d)}
NAME=morie-339-audit-$STAMP

echo "== 1. zeus: git mirror =="
git -C "$ROOT/morie" push -q zeus main 2>&1 | grep -viE 'lfs|locksverify'
echo "   mirror HEAD: $(ssh zeus "git -C /mnt/nvme/git-mirror/morie.git rev-parse --short main")"
echo "   local  HEAD: $(git -C "$ROOT/morie" rev-parse --short HEAD)"

echo "== 2. zeus: GRAD_STUDIES =="
rsync -a "$ROOT/GRAD_STUDIES/" zeus:/mnt/nvme/GRAD_WORK/backups/GRAD_STUDIES/ && echo "   ok"

echo "== 3. L14 internal staging =="
ssh l14 "mkdir -p ~/backups/$NAME"
rsync -a "$ROOT/GRAD_STUDIES"/morie_339_audit_*.txt   "l14:~/backups/$NAME/"
rsync -a "$ROOT/morie-audit-artifacts/"               "l14:~/backups/$NAME/artifacts/"
rsync -a "$ROOT/morie/tests/fn/_audit/"               "l14:~/backups/$NAME/audit/"
rsync -a "$ROOT/morie/tests/fn/fixtures/"             "l14:~/backups/$NAME/fixtures/"
echo "   $(ssh l14 "du -sh ~/backups/$NAME | cut -f1")"

echo "== 4. WD_BLACK (located, not assumed) =="
WD=$(ssh l14 "findmnt -n -o TARGET -S LABEL=WD_BLACK 2>/dev/null | head -1")
if [ -z "$WD" ]; then
  echo "   WD_BLACK not mounted -- three copies stand, WD is the fourth."
  echo "   To attach: plug in, then  findmnt -S LABEL=WD_BLACK"
  exit 0
fi
echo "   found at: $WD"
ssh l14 "mkdir -p '$WD/$NAME' && rsync -a ~/backups/$NAME/ '$WD/$NAME/'" || { echo "   rsync FAILED"; exit 1; }

echo "== 5. verify WD copy by checksum =="
ssh l14 "cd ~/backups/$NAME && find . -type f -exec sha256sum {} \; | sort > /tmp/a.sums
         cd '$WD/$NAME'      && find . -type f -exec sha256sum {} \; | sort > /tmp/b.sums
         if diff -q /tmp/a.sums /tmp/b.sums >/dev/null; then
           echo \"   ALL FILES IDENTICAL (\$(wc -l < /tmp/a.sums) verified)\"
         else
           echo '   MISMATCH:'; diff /tmp/a.sums /tmp/b.sums; exit 1
         fi"
