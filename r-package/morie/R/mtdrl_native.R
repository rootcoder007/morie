# mtdrl -- deep meta-reinforcement learning (Wang et al. 2016)
# Reference: Wang et al. (2016) "Learning to reinforcement learn" arXiv:1611.05763
# Base R only.

mtdrl_bandit_tasks <- function(n_arms = 2, n_tasks = 100, seed = 0,
                               structure = "independent") {
  if (!(structure %in% c("independent", "paired"))) {
    stop(sprintf("mtdrl: structure must be 'independent' or 'paired', got %s", structure))
  }
  n_arms <- as.integer(n_arms)
  if (n_arms < 2L) stop("mtdrl: need at least 2 arms")
  if (structure == "paired" && n_arms != 2L) {
    stop("mtdrl: the paired family is defined for 2 arms")
  }
  set.seed(seed)
  tasks <- list()
  for (i in seq_len(as.integer(n_tasks))) {
    if (structure == "paired") {
      p <- runif(1)
      tasks[[length(tasks) + 1L]] <- c(p, 1 - p)
    } else {
      tasks[[length(tasks) + 1L]] <- runif(n_arms)
    }
  }
  tasks
}

mtdrl_history_features <- function(history, n_arms) {
  feat <- rep(0, n_arms + 2L)
  if (length(history) > 0L) {
    last <- history[[length(history)]]
    feat[last[[1]] + 1L] <- 1
    feat[n_arms + 1L] <- as.numeric(last[[2]])
  }
  feat[n_arms + 2L] <- length(history)
  feat
}

mtdrl_TabularHistoryAgent <- function(n_arms, epsilon = 0.1, optimistic = 1) {
  agent <- new.env(parent = emptyenv())
  agent$n_arms <- as.integer(n_arms)
  agent$epsilon <- as.numeric(epsilon)
  agent$optimistic <- as.numeric(optimistic)
  agent$counts <- rep(0L, agent$n_arms)
  agent$means <- rep(agent$optimistic, agent$n_arms)
  agent$reset <- function() {
    agent$counts <- rep(0L, agent$n_arms)
    agent$means <- rep(agent$optimistic, agent$n_arms)
  }
  agent$act <- function(features, rng) {
    if (rng() < agent$epsilon) {
      return(sample.int(agent$n_arms, 1L) - 1L)
    }
    best <- max(agent$means)
    cand <- which(agent$means >= best)
    cand[as.integer(rng() * length(cand)) + 1L] - 1L
  }
  agent$observe <- function(action, reward) {
    a <- as.integer(action) + 1L
    agent$counts[a] <- agent$counts[a] + 1L
    n <- agent$counts[a]
    agent$means[a] <- agent$means[a] + (as.numeric(reward) - agent$means[a]) / n
  }
  agent
}

mtdrl_run <- function(tasks, agent, episode_length = 100, n_arms = NULL,
                      seed = 0, reset_between_episodes = TRUE) {
  T_ <- lapply(tasks, as.numeric)
  if (length(T_) == 0L) stop("mtdrl: tasks must be non-empty")
  k <- if (is.null(n_arms)) length(T_[[1]]) else as.integer(n_arms)
  for (t in T_) {
    if (length(t) != k) stop(sprintf("mtdrl: every task must have %d arms", k))
  }
  L <- as.integer(episode_length)
  if (L < 1L) stop("mtdrl: episode_length must be >= 1")
  for (m in c("reset", "act", "observe")) {
    if (!exists(m, envir = agent, inherits = FALSE)) {
      stop(sprintf("mtdrl: agent must provide %s()", m))
    }
  }
  set.seed(seed)
  rng <- function() runif(1)
  total <- 0
  regret <- 0
  by_step <- rep(0, L)
  opt_by_step <- rep(0, L)
  per_episode <- c()
  for (probs in T_) {
    if (reset_between_episodes) agent$reset()
    best_p <- max(probs)
    best_arms <- which(probs >= best_p)
    hist <- list()
    ep_reward <- 0
    for (t in seq_len(L)) {
      feats <- mtdrl_history_features(hist, k)
      a <- as.integer(agent$act(feats, rng))
      if (!(a >= 0 && a < k)) {
        stop(sprintf("mtdrl: agent chose arm %d outside 0..%d", a, k - 1L))
      }
      r <- if (runif(1) < probs[a + 1L]) 1 else 0
      agent$observe(a, r)
      hist[[length(hist) + 1L]] <- list(a, r)
      ep_reward <- ep_reward + r
      total <- total + r
      regret <- regret + (best_p - probs[a + 1L])
      by_step[t] <- by_step[t] + r
      opt_by_step[t] <- opt_by_step[t] + (if (a %in% best_arms) 1 else 0)
    }
    per_episode <- c(per_episode, ep_reward)
  }
  n_ep <- as.numeric(length(T_))
  list(estimate = total / (n_ep * L),
       mean_reward = total / (n_ep * L),
       total_reward = total,
       regret = regret,
       reward_by_step = by_step / n_ep,
       optimal_action_rate = opt_by_step / n_ep,
       episode_reward = per_episode,
       n_episodes = length(T_),
       episode_length = L,
       n_arms = k,
       method = "meta-RL evaluation loop (Wang et al. 2016 sec. 2)")
}

mtdrl_cheatsheet <- function() {
  paste("mtdrl: deep meta-RL (Wang 2016). Train with one RL algorithm so the RECURRENT DYNAMICS implement a second, learned one. Policy conditions on the whole within-episode history H_t including the previous ACTION and REWARD; the recurrent state is RESET each episode, and after training the weights are frozen so all within-episode adaptation is in the activations. bandit_tasks(structure='paired') is the dependent-arm family whose structure an adapted inner algorithm can exploit.")
}

# house entry point: the package exports one morie_<module>
morie_mtdrl <- mtdrl_bandit_tasks
