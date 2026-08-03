"""Collapse byte-identical MVSML modules into one canonical
implementation plus thin re-export shims.

The auto-extractor stamped many different page fragments with the same
function name, so several files ended up carrying an identical copy of
the same book equation.  Keep ONE real implementation per (function
name, equation) and turn the rest into shims that import it, so the
stub-name contract still resolves but the body exists once.
"""
import hashlib, glob, re, sys
from pathlib import Path

groups = {}
for p in sorted(glob.glob("src/morie/fn/msm*.py")):
    s = Path(p).read_text()
    if "result = float(np.mean(" in s:
        continue
    if "re-exported from" in s:
        continue
    mod = Path(p).stem
    key = hashlib.md5(s.replace(mod, "<MOD>").encode()).hexdigest()
    groups.setdefault(key, []).append((mod, s))

shimmed = 0
for key, members in groups.items():
    if len(members) < 2:
        continue
    members.sort(key=lambda t: t[0])
    canon_mod, canon_src = members[0]
    fn = re.search(r'__all__ = \["([a-z_0-9]+)"', canon_src).group(1)
    cite = re.search(r'Implements (.+?) of Montesinos', canon_src,
                     re.S)
    cite = " ".join(cite.group(1).split()) if cite else "the model"
    for mod, _ in members[1:]:
        shim = (
            '# morie.fn -- function file (rootcoder007/morie)\n'
            '"""%s, re-exported from :mod:`morie.fn.%s`.\n\n'
            'The stub generator stamped several extracted page\n'
            'fragments with this same function name, so the\n'
            'implementation lives once in %s and this module re-exports\n'
            'it.  Calling either path runs the same code.\n"""\n\n'
            'from .%s import %s\n\n'
            '__all__ = ["%s"]\n\n\n'
            'def cheatsheet():\n'
            '    return "%s: see %s"\n'
            % (cite, canon_mod, canon_mod, canon_mod, fn, fn, mod,
               canon_mod))
        Path("src/morie/fn/%s.py" % mod).write_text(shim)
        shimmed += 1
print("canonical implementations kept:", 
      sum(1 for m in groups.values() if m))
print("files converted to re-export shims:", shimmed)
