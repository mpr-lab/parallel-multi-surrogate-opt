"""
GP refresh scheduler — controls when the expensive surrogate is retrained.

This implements the 'cost-aware GP refresh' idea at the heart of the prototype:
instead of retraining the GP every round, we only retrain it every K rounds.

The tradeoff:
  - K=1:  GP is always fresh.  Expensive (full O(N^3) fit per round).
  - K=5:  GP is retrained every 5 rounds.  Cheaper, slight staleness.
  - K=10: GP is retrained every 10 rounds.  Very cheap, but proposals
          from the GP use increasingly outdated information.

The 'fast' surrogate (RF) always retrains every round regardless of K.

This is deliberately simple — no adaptive policy, just a fixed period.
Future work could make K depend on the GP fit time or improvement rate.
"""


class GPRefreshScheduler:
    """
    Decides whether the GP should be retrained in the current round.

    Usage:
        scheduler = GPRefreshScheduler(refresh_every=5)
        for round_id in range(n_rounds):
            if scheduler.should_refresh_gp(round_id):
                gp.fit(X, y)
    """

    def __init__(self, refresh_every: int):
        """
        Args:
            refresh_every: Retrain the GP every this many rounds.
                           refresh_every=1  -> retrain every round (baseline)
                           refresh_every=5  -> retrain rounds 0, 5, 10, 15, ...
                           refresh_every=10 -> retrain rounds 0, 10, 20, ...

        Raises:
            ValueError if refresh_every < 1.
        """
        if refresh_every < 1:
            raise ValueError(
                f"refresh_every must be >= 1, got {refresh_every}. "
                "Use refresh_every=1 to retrain every round."
            )
        self.refresh_every = refresh_every

    def should_refresh_gp(self, round_id: int) -> bool:
        """
        Return True if the GP should be retrained in round `round_id`.

        Round 0 always triggers a refresh so the GP is trained at least once.

        Args:
            round_id: Current round number (0-indexed).

        Returns:
            True  -> retrain the GP this round.
            False -> reuse the existing (stale) GP model from the last refresh.
        """
        # round_id % refresh_every == 0 covers round 0 and every Kth round after
        return (round_id % self.refresh_every) == 0

    def __repr__(self) -> str:
        return f"GPRefreshScheduler(refresh_every={self.refresh_every})"
