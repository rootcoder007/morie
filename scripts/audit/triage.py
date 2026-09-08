import ast, pathlib, re, collections, json
FN = pathlib.Path("src/morie/fn")
STUB = re.compile(r"np\.std\(.*ddof=1\) / np\.sqrt\(n\)|stats\.spearmanr|x_sorted = np\.sort\(X\)")

DOMAINS = [
 ("causal-inference", r"causal|treatment effect|propensity|instrument|did |difference-in-diff|synthetic control|mediation|iv |rdd|regression discontinuity|doubly robust|matching"),
 ("time-series",      r"garch|arima|arma|var\(|vecm|cointegrat|kalman|state.space|filter|seasonal|forecast|autocorrel|unit root|spectral|wavelet|hurst|volatilit"),
 ("spatial",          r"spatial|kriging|variogram|moran|geary|ripley|point pattern|geostat|areal|lattice|gwr|lisa"),
 ("psychometrics",    r"irt|item response|cronbach|factor analys|reliab|dif |differential item|rasch|sem |structural equation|latent"),
 ("survival",         r"survival|hazard|kaplan|cox |censor|time.to.event|frailty|competing risk"),
 ("ml-deep",          r"neural|transformer|attention|embedding|gradient descent|adam|dropout|convolution|lstm|gru|encoder|decoder|gan |vae|diffusion|rlhf|token"),
 ("ml-classical",     r"random forest|boost|svm|kernel|cluster|k-means|pca|lda |naive bayes|knn|ensemble|cross.valid|regulariz|lasso|ridge|elastic"),
 ("bayesian",         r"bayes|posterior|prior|mcmc|gibbs|metropolis|variational|hamiltonian|dirichlet process|abc "),
 ("epidemiology",     r"epidemi|incidence|prevalence|outbreak|contagion|sir |seir|vaccin|screening|odds ratio|relative risk"),
 ("genomics",         r"genom|gwas|snp|allele|haplotype|heritab|linkage|sequencing|expression"),
 ("signal-imaging",   r"eeg|fmri|ecg|signal|fourier|spectrogram|denois|image|voxel|entropy|fractal|lyapunov"),
 ("econometrics",     r"panel|fixed effect|random effect|gmm|two.stage|heteroskedast|endogen|elasticity|production function"),
 ("survey-sampling",  r"survey|weight|stratif|cluster sampl|raking|post.strat|nonresponse|horvitz"),
 ("nonparametric",    r"nonparametr|rank |permutation|bootstrap|jackknife|kernel density|quantile|wilcoxon|kruskal|spline"),
 ("network",          r"network|graph|centrality|community detect|edge |node |adjacency|small.world"),
]
rows = []
for f in sorted(FN.glob("*.py")):
    if f.name.startswith("_") or f.name in ("describe.py",):
        continue
    try:
        src = f.read_text()
    except Exception:
        continue
    if not STUB.search(src):
        continue
    doc = (ast.get_docstring(ast.parse(src)) or "").lower()
    head = doc.splitlines()[0] if doc else ""
    dom = "other"
    for name, pat in DOMAINS:
        if re.search(pat, doc, re.I):
            dom = name
            break
    rows.append((dom, f.stem, head[:70]))
cnt = collections.Counter(r[0] for r in rows)
print(f"placeholders classified: {len(rows)}\n")
for d, c in cnt.most_common():
    print(f"  {d:18s} {c:6d}  {100*c/len(rows):5.1f}%")
json.dump([{"domain": d, "module": m, "title": t} for d, m, t in rows],
          open("/tmp/triage.json", "w"), indent=0)
